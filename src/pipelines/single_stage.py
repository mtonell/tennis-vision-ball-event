import os
import time
import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader

from src.models.dl import get_dl_model
from src.models.ml import get_ml_model
from src.data.dataset import TrajectoryDataset
from src.utils.metrics import apply_nms_and_tolerance, evaluate_and_log_results, get_event_macro_metric
from src.training.trainer import train_dl_stage, train_ml_stage

def run_single_stage_pipeline(args, df_train, df_val, df_test, X_train, X_val, X_test, model_family, device, experiment, run_name):
    y_train = df_train['target_multi'].values
    y_val = df_val['target_multi'].values if len(df_val) > 0 else []
    y_test = df_test['target_multi'].values if len(df_test) > 0 else []
    num_classes = 3
    
    if model_family == 'dl':
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = torch.FloatTensor(cw).to(device)
        
        pin_mem = device.type == 'cuda'
        train_loader = DataLoader(TrajectoryDataset(X_train, y_train, augment=True), batch_size=64, shuffle=True, pin_memory=pin_mem)
        val_loader = DataLoader(TrajectoryDataset(X_val, y_val, augment=False), batch_size=64, shuffle=False, pin_memory=pin_mem) if len(X_val) > 0 else None
        
        in_channels = 6 if args.data == 'pva' else 2
        model = get_dl_model(args.model, args.window, in_channels, num_classes).to(device)
        train_dl_stage(model, train_loader, val_loader, args.epochs, device, experiment, "Stage 1", class_weights, args.loss, lr=args.lr, weight_decay=args.weight_decay, early_stopping_metric=args.metric, df_val=df_val, nms_window=args.nms)
        
        if not args.deploy:
            model.eval()
            with torch.no_grad():
                if len(X_val) > 0:
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                    probs_val = torch.softmax(model(X_val_tensor), dim=1).cpu().numpy()
                else:
                    probs_val = None
                    
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                start_time = time.time()
                logits = model(X_tensor)
                probs_test = torch.softmax(logits, dim=1).cpu().numpy()
                inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
    else:
        model = get_ml_model(args.model, num_classes)
        model = train_ml_stage(model, X_train, y_train, "SingleStage")
        
        if not args.deploy:
            probs_val = model.predict_proba(X_val) if len(X_val) > 0 else None
            start_time = time.time()
            probs_test = model.predict_proba(X_test)
            inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
        
    if args.save:
        os.makedirs('weights', exist_ok=True)
        if model_family == 'dl':
            torch.save(model.state_dict(), f"weights/{run_name}.pth")
            print(f"Model weights saved to weights/{run_name}.pth")
        else:
            joblib.dump(model, f"weights/{run_name}.pkl")
            print(f"Model weights saved to weights/{run_name}.pkl")
            
    if args.deploy:
        print(f"\nDeployment model successfully trained on ALL data for {args.epochs} epochs!")
        if experiment: experiment.end()
        return
        
    print(f"\n=== Single Stage Test Results ===")
    
    # Tune on validation set
    best_val_thresh = 0.0
    if probs_val is not None and len(probs_val) > 0:
        best_val_f1 = -1
        for thresh in np.linspace(0.1, 0.95, 30):
            preds_val = np.argmax(probs_val, axis=1)
            max_probs_val = np.max(probs_val, axis=1)
            preds_val[(preds_val != 0) & (max_probs_val < thresh)] = 0
            if args.nms > 0:
                preds_val = apply_nms_and_tolerance(y_val, preds_val, max_probs_val, df_val, args.nms)
            f1 = get_event_macro_metric(y_val, preds_val, metric='f1')
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_val_thresh = thresh
        print(f"Optimal Confidence Threshold on Val: {best_val_thresh:.3f} (Max Val F1: {best_val_f1:.4f})")
    
    # Evaluate on test set
    max_probs_test = np.max(probs_test, axis=1)
    
    # Baseline
    preds_base = np.argmax(probs_test, axis=1)
    if args.nms > 0:
        preds_base = apply_nms_and_tolerance(y_test, preds_base, max_probs_test, df_test, args.nms)
        
    # Tuned
    preds_tuned = np.argmax(probs_test, axis=1)
    preds_tuned[(preds_tuned != 0) & (max_probs_test < best_val_thresh)] = 0
    if args.nms > 0:
        preds_tuned = apply_nms_and_tolerance(y_test, preds_tuned, max_probs_test, df_test, args.nms)
        
    # Oracle
    best_test_thresh = 0.0
    best_test_f1 = -1
    best_preds_oracle = None
    for thresh in np.linspace(0.1, 0.95, 30):
        preds_oracle = np.argmax(probs_test, axis=1)
        preds_oracle[(preds_oracle != 0) & (max_probs_test < thresh)] = 0
        if args.nms > 0:
            preds_oracle = apply_nms_and_tolerance(y_test, preds_oracle, max_probs_test, df_test, args.nms)
        f1 = get_event_macro_metric(y_test, preds_oracle, metric='f1')
        if f1 > best_test_f1:
            best_test_f1 = f1
            best_preds_oracle = preds_oracle
            
    preds_dict = {
        'Baseline': preds_base,
        'Tuned': preds_tuned,
        'Oracle': best_preds_oracle
    }
        
    evaluate_and_log_results(y_test, preds_dict, inference_time_ms, experiment)
