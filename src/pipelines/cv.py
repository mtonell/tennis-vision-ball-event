import os
import time
import joblib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import load_data, extract_features, TrajectoryDataset
from src.models.dl import get_dl_model
from src.models.ml import get_ml_model
from src.utils.metrics import apply_nms_and_tolerance, get_event_macro_metric, evaluate_and_log_results
from src.utils.config import DATASET_GAMES
from src.training.trainer import train_dl_stage, train_ml_stage
from sklearn.metrics import precision_recall_curve

def run_cv_pipeline(args, device, experiment, dataset_path, nms_window=3, model_family='dl'):
    in_channels = 6 if args.data == 'pva' else 2
    max_epochs = args.epochs
    
    cv_pool = DATASET_GAMES['train'] + DATASET_GAMES['val']  # Games 1-8
    
    folds = [
        {'val': cv_pool[:2], 'train': cv_pool[2:]},
        {'val': cv_pool[2:4], 'train': cv_pool[:2] + cv_pool[4:]},
        {'val': cv_pool[4:6], 'train': cv_pool[:4] + cv_pool[6:]},
        {'val': cv_pool[6:], 'train': cv_pool[:6]}
    ]

    oof_probs = []
    oof_labels = []
    oof_dfs = []
    stopping_epochs = []

    print(f"\nSTARTING PHASE 1 CV: {args.model.upper()} | Window: {args.window} | Metric: {args.metric.upper()}")

    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Fold {fold_idx+1}/4 ---")
        df_train, df_val, _ = load_data(dataset_path, args.window, model_family, balance=True, 
                                        train_games=fold['train'], val_games=fold['val'], test_games=[])
                                        
        y_train = df_train['target_multi'].values
        y_val = df_val['target_multi'].values
        
        X_train = extract_features(df_train, args.window, model_family, data_type=args.data)
        X_val = extract_features(df_val, args.window, model_family, data_type=args.data)
        
        if model_family == 'dl':
            cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights = torch.FloatTensor(cw).to(device)
            
            train_loader = DataLoader(TrajectoryDataset(X_train, y_train, augment=True), batch_size=64, shuffle=True)
            val_loader = DataLoader(TrajectoryDataset(X_val, y_val, augment=False), batch_size=64, shuffle=False)
            
            model = get_dl_model(args.model, args.window, in_channels, num_classes=3).to(device)
            
            best_epoch, best_weights = train_dl_stage(
                model, train_loader, val_loader, max_epochs, device, experiment, f"Fold {fold_idx+1}", 
                class_weights=class_weights, loss_type=args.loss, lr=args.lr, weight_decay=args.weight_decay,
                early_stopping_metric=args.metric, df_val=df_val, nms_window=nms_window
            )
            
            stopping_epochs.append(best_epoch)
            if experiment:
                experiment.log_metric(f"fold{fold_idx+1}_stopping_epoch", best_epoch)
            
            model.load_state_dict(best_weights)
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            model = get_ml_model(args.model, num_classes=3)
            model = train_ml_stage(model, X_train, y_train, stage_name=f"Fold {fold_idx+1}")
            probs = model.predict_proba(X_val)
            
        oof_probs.append(probs)
        oof_labels.append(y_val)
        oof_dfs.append(df_val)

    if model_family == 'dl':
        mean_epochs = int(np.mean(stopping_epochs))
        print(f"\n--- CV Complete! Mean Stopping Epochs: {mean_epochs} ---")
        if experiment:
            experiment.log_metric("mean_epochs", mean_epochs)
    else:
        mean_epochs = 0
        
    all_oof_probs = np.concatenate(oof_probs)
    all_oof_labels = np.concatenate(oof_labels)
    all_oof_df = pd.concat(oof_dfs, ignore_index=True)
    
    max_probs_oof = np.max(all_oof_probs, axis=1)
    
    if experiment:
        prob_hit = all_oof_probs[:, 1]
        prob_bounce = all_oof_probs[:, 2]
        y_true_hit = (all_oof_labels == 1).astype(int)
        y_true_bounce = (all_oof_labels == 2).astype(int)
        prec_hit, rec_hit, _ = precision_recall_curve(y_true_hit, prob_hit)
        prec_bounce, rec_bounce, _ = precision_recall_curve(y_true_bounce, prob_bounce)
        experiment.log_curve("OOF_PR_Curve_Hit", rec_hit.tolist(), prec_hit.tolist())
        experiment.log_curve("OOF_PR_Curve_Bounce", rec_bounce.tolist(), prec_bounce.tolist())
    
    best_oof_thresh = 0.0
    best_oof_f1 = -1
    
    print("\n--- Sweeping Threshold on OOF Predictions ---")
    for thresh in np.linspace(0.1, 0.95, 30):
        preds = np.argmax(all_oof_probs, axis=1)
        preds[(preds != 0) & (max_probs_oof < thresh)] = 0
        preds = apply_nms_and_tolerance(all_oof_labels, preds, max_probs_oof, all_oof_df, nms_window)
        f1 = get_event_macro_metric(all_oof_labels, preds, metric='f1')
        if f1 > best_oof_f1:
            best_oof_f1 = f1
            best_oof_thresh = thresh
            
    print(f">> BEST OOF THRESHOLD: {best_oof_thresh:.3f} (OOF F1: {best_oof_f1:.4f})")
    if experiment:
        experiment.log_metric("oof_peak_threshold", best_oof_thresh)
        experiment.log_metric("oof_peak_f1", best_oof_f1)
        
    print(f"\n--- Training Final Phase 1 Model on Games 1-8 for {mean_epochs if model_family == 'dl' else 'Full'} Epochs ---")
    df_train_full, _, df_test = load_data(
        dataset_path, args.window, model_family, balance=True,
        train_games=['game1','game2','game3','game4','game5','game6','game7','game8'], 
        val_games=[], test_games=['game9','game10']
    )
    y_train_full = df_train_full['target_multi'].values
    X_train_full = extract_features(df_train_full, args.window, model_family, data_type=args.data)
    y_test = df_test['target_multi'].values
    X_test = extract_features(df_test, args.window, model_family, data_type=args.data)
    
    if model_family == 'dl':
        cw_full = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)
        class_weights_full = torch.FloatTensor(cw_full).to(device)
        
        train_loader_full = DataLoader(TrajectoryDataset(X_train_full, y_train_full, augment=True), batch_size=64, shuffle=True)
        model_final = get_dl_model(args.model, args.window, in_channels, num_classes=3).to(device)
        
        # Train Final Validation model (returns best weights from last epoch)
        _, best_final_weights = train_dl_stage(
            model_final, train_loader_full, None, mean_epochs, device, experiment, "Phase 1 Final", 
            class_weights=class_weights_full, loss_type=args.loss, lr=args.lr, weight_decay=args.weight_decay
        )
        model_final.load_state_dict(best_final_weights)
                
        print("\n--- Final Test Set Evaluation (Games 9-10) ---")
        model_final.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            start_time = time.time()
            logits_test = model_final(X_test_tensor)
            inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
            probs_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    else:
        model_final = get_ml_model(args.model, num_classes=3)
        model_final = train_ml_stage(model_final, X_train_full, y_train_full, stage_name="Phase 1 Final")
        
        print("\n--- Final Test Set Evaluation (Games 9-10) ---")
        start_time = time.time()
        probs_test = model_final.predict_proba(X_test)
        inference_time_ms = (time.time() - start_time) * 1000 / len(X_test) if len(X_test) > 0 else 0
    # Baseline
    max_probs_test = np.max(probs_test, axis=1)
    preds_test_base = np.argmax(probs_test, axis=1)
    preds_test_base = apply_nms_and_tolerance(y_test, preds_test_base, max_probs_test, df_test, nms_window)
    
    # Tuned
    preds_test_tuned = np.argmax(probs_test, axis=1)
    preds_test_tuned[(preds_test_tuned != 0) & (max_probs_test < best_oof_thresh)] = 0
    preds_test_tuned = apply_nms_and_tolerance(y_test, preds_test_tuned, max_probs_test, df_test, nms_window)
    
    # Oracle (Best theoretical threshold for the Test Set)
    best_test_thresh = 0.0
    best_test_f1 = -1
    best_preds_oracle = None
    for thresh in np.linspace(0.1, 0.95, 30):
        preds = np.argmax(probs_test, axis=1)
        preds[(preds != 0) & (max_probs_test < thresh)] = 0
        preds = apply_nms_and_tolerance(y_test, preds, max_probs_test, df_test, nms_window)
        f1 = get_event_macro_metric(y_test, preds, metric='f1')
        if f1 > best_test_f1:
            best_test_f1 = f1
            best_test_thresh = thresh
            best_preds_oracle = preds
    
    if getattr(args, 'save', False):
        os.makedirs('weights', exist_ok=True)
        if model_family == 'dl':
            torch.save(model_final.state_dict(), f"weights/{args.model}_phase1_cv_win{args.window}.pth")
        else:
            joblib.dump(model_final, f"weights/{args.model}_phase1_cv_win{args.window}.pkl")
            
    preds_dict = {
        'Baseline': preds_test_base,
        'Tuned': preds_test_tuned,
        'Oracle': best_preds_oracle
    }
    
    evaluate_and_log_results(y_test, preds_dict, inference_time_ms, experiment)
