import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.models.dl import get_dl_model
from src.models.ml import get_ml_model
from src.data.dataset import TrajectoryDataset
from src.utils.metrics import apply_nms_and_tolerance, evaluate_and_log_results
from src.training.trainer import train_dl_stage, train_ml_stage

def run_multi_stage_pipeline(args, df_train, df_val, df_test, X_train, X_val, X_test, model_family, device, experiment):
    if args.deploy:
        print("\nDeploy mode is not supported for multi-stage pipelines. Use single-stage with --stages 1.")
        if experiment: experiment.end()
        return
    
    print("--- Training Stage 1 (Anomaly: Flying vs Event) ---")
    y_train_s1 = df_train['target_binary'].values
    y_val_s1 = df_val['target_binary'].values
    
    if model_family == 'dl':
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_s1)
        cw = compute_class_weight('balanced', classes=classes, y=y_train_s1)
        class_weights = torch.FloatTensor(cw).to(device)
        
        train_loader_s1 = DataLoader(TrajectoryDataset(X_train, y_train_s1, augment=True), batch_size=64, shuffle=True)
        val_loader_s1 = DataLoader(TrajectoryDataset(X_val, y_val_s1), batch_size=64, shuffle=False)
        
        in_channels = 6 if args.data == 'pva' else 2
        model_s1 = get_dl_model(args.model, args.window, in_channels, num_classes=2).to(device)  # Binary: flying vs event
        train_dl_stage(model_s1, train_loader_s1, val_loader_s1, args.epochs, device, experiment, "Stage1", class_weights, loss_type=args.loss, early_stopping_metric='auc' if args.metric == 'auc' else args.metric, eval_metric='f2', df_val=df_val, nms_window=args.nms, lr=args.lr, weight_decay=args.weight_decay)
    else:
        model_s1 = get_ml_model(args.model, 2)
        model_s1 = train_ml_stage(model_s1, X_train, y_train_s1, "Stage1")
        
    print("--- Training Stage 2 (Geometry: Hit vs Bounce) ---")
    mask_train = df_train['target_event'] != -1
    mask_val = df_val['target_event'] != -1
    X_train_s2 = X_train[mask_train]
    y_train_s2 = df_train['target_event'].values[mask_train]
    X_val_s2 = X_val[mask_val]
    y_val_s2 = df_val['target_event'].values[mask_val]
    
    if model_family == 'dl':
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_s2)
        cw = compute_class_weight('balanced', classes=classes, y=y_train_s2)
        class_weights = torch.FloatTensor(cw).to(device)
        
        train_loader_s2 = DataLoader(TrajectoryDataset(X_train_s2, y_train_s2, augment=False), batch_size=32, shuffle=True)
        val_loader_s2 = DataLoader(TrajectoryDataset(X_val_s2, y_val_s2), batch_size=32, shuffle=False)
        
        in_channels = 6 if args.data == 'pva' else 2
        model_s2 = get_dl_model(args.model, args.window, in_channels, num_classes=2).to(device)  # Binary: hit vs bounce
        df_val_s2 = df_val[mask_val].copy().reset_index(drop=True)
        train_dl_stage(model_s2, train_loader_s2, val_loader_s2, args.epochs, device, experiment, "Stage2", class_weights, loss_type=args.loss, early_stopping_metric='auc' if args.metric == 'auc' else args.metric, eval_metric='f1', df_val=df_val_s2, nms_window=0, lr=args.lr, weight_decay=args.weight_decay)
        
        model_s1.eval()
        model_s2.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            probs_val_s1 = torch.softmax(model_s1(X_val_tensor), dim=1)[:, 1].cpu().numpy()
            probs_val_s2_bounce = torch.softmax(model_s2(X_val_tensor), dim=1)[:, 1].cpu().numpy()  # P(Bounce) from stage 2
            
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            start_time = time.time()
            probs_test_s1 = torch.softmax(model_s1(X_tensor), dim=1)[:, 1].cpu().numpy()
            probs_test_s2 = torch.softmax(model_s2(X_tensor), dim=1)[:, 1].cpu().numpy()
            inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
    else:
        model_s2 = get_ml_model(args.model, 2)
        model_s2 = train_ml_stage(model_s2, X_train_s2, y_train_s2, "Stage2")
        
        probs_val_s1 = model_s1.predict_proba(X_val)[:, 1]
        probs_val_s2_bounce = model_s2.predict_proba(X_val)[:, 1]  # P(Bounce) from stage 2
        
        start_time = time.time()
        probs_test_s1 = model_s1.predict_proba(X_test)[:, 1]
        probs_test_s2 = model_s2.predict_proba(X_test)[:, 1]
        inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
        
    # -------------------------------------------------------------
    # Tune Soft-Passing Confidence T on Validation Set
    # -------------------------------------------------------------
    print("\n--- Tuning Soft-Passing Confidence T on Validation Set ---")
    probs_val_flying = 1.0 - probs_val_s1
    probs_val_bounce = probs_val_s1 * probs_val_s2_bounce
    probs_val_hit = probs_val_s1 * (1.0 - probs_val_s2_bounce)
        
    joint_probs_val = np.vstack([probs_val_flying, probs_val_hit, probs_val_bounce]).T
    val_event_max_probs = np.max(joint_probs_val[:, 1:], axis=1)
    val_event_argmax = np.argmax(joint_probs_val[:, 1:], axis=1) + 1
        
    best_soft_thresh = 0.5
    best_soft_val_f1 = -1
        
    for t in np.linspace(0.1, 0.95, 50):
        preds_val_soft = np.where(val_event_max_probs >= t, val_event_argmax, 0)
        if args.nms > 0:
            preds_val_soft = apply_nms_and_tolerance(df_val['target_multi'].values, preds_val_soft, probs_val_s1, df_val, args.nms)
        report = classification_report(df_val['target_multi'].values, preds_val_soft, labels=[0, 1, 2], output_dict=True, zero_division=0)
        f1 = (report['1']['f1-score'] + report['2']['f1-score']) / 2.0
        if f1 > best_soft_val_f1:
            best_soft_val_f1 = f1
            best_soft_thresh = t
                
    print(f"Optimal Confidence Threshold: {best_soft_thresh:.4f} (Max Val F1: {best_soft_val_f1:.4f})")
    
    # -------------------------------------------------------------
    # Calculate Bayes Soft-Passing (Zero-Parameter Probability Cascade)
    # -------------------------------------------------------------
    probs_test_flying = 1.0 - probs_test_s1
    probs_test_bounce = probs_test_s1 * probs_test_s2
    probs_test_hit = probs_test_s1 * (1.0 - probs_test_s2)
    
    joint_probs_test = np.vstack([probs_test_flying, probs_test_hit, probs_test_bounce]).T
    test_event_max_probs = np.max(joint_probs_test[:, 1:], axis=1)
    test_event_argmax = np.argmax(joint_probs_test[:, 1:], axis=1) + 1
    
    y_test_final = df_test['target_multi'].values
    
    # 1. Standard (Baseline Argmax)
    preds_soft_base = np.argmax(joint_probs_test, axis=1)
    if args.nms > 0:
        preds_soft_base = apply_nms_and_tolerance(y_test_final, preds_soft_base, probs_test_s1, df_test, args.nms)
    report_base = classification_report(y_test_final, preds_soft_base, labels=[0, 1, 2], output_dict=True, zero_division=0)
    
    # 2. Tuned (Validation Threshold)
    preds_soft_tuned = np.where(test_event_max_probs >= best_soft_thresh, test_event_argmax, 0)
    if args.nms > 0:
        preds_soft_tuned = apply_nms_and_tolerance(y_test_final, preds_soft_tuned, probs_test_s1, df_test, args.nms)
    report_tuned = classification_report(y_test_final, preds_soft_tuned, labels=[0, 1, 2], output_dict=True, zero_division=0)
    soft_tuned_event_f1 = (report_tuned['1']['f1-score'] + report_tuned['2']['f1-score']) / 2.0
    
    # 3. Max (Oracle Test Threshold)
    best_soft_test_f1 = -1
    best_preds_oracle = None
    for t in np.linspace(0.1, 0.95, 50):
        preds_test_soft_oracle = np.where(test_event_max_probs >= t, test_event_argmax, 0)
        if args.nms > 0:
            preds_test_soft_oracle = apply_nms_and_tolerance(y_test_final, preds_test_soft_oracle, probs_test_s1, df_test, args.nms)
        report_oracle = classification_report(y_test_final, preds_test_soft_oracle, labels=[0, 1, 2], output_dict=True, zero_division=0)
        f1 = (report_oracle['1']['f1-score'] + report_oracle['2']['f1-score']) / 2.0
        if f1 > best_soft_test_f1:
            best_soft_test_f1 = f1
            best_preds_oracle = preds_test_soft_oracle

    preds_dict = {
        'Baseline': preds_soft_base,
        'Tuned': preds_soft_tuned,
        'Oracle': best_preds_oracle
    }
    
    evaluate_and_log_results(y_test_final, preds_dict, inference_time_ms, experiment)
    
    return soft_tuned_event_f1
