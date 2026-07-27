import argparse
import yaml
import optuna
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import load_data, extract_features, TrajectoryDataset
from src.models.ml import get_ml_model
from src.models.dl import get_dl_model
from src.training.trainer import train_dl_stage
from src.utils.metrics import get_event_macro_metric_curve_area
from src.utils.config import DL_MODELS, ML_MODELS, DATASET_GAMES

def objective(trial, args, data_cache, param_space, model_family):
    params = {}
    for param_name, config in param_space.items():
        if config['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, config['low'], config['high'])
        elif config['type'] == 'float':
            log_scale = config.get('log', False)
            params[param_name] = trial.suggest_float(param_name, config['low'], config['high'], log=log_scale)
        elif config['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, config['choices'])
            
    if 'window' not in params:
        raise KeyError(f"'window' is required in tune_parameters.yaml for model '{args.model}' but was not found.")
    window = params.pop('window')
    
    if model_family == 'dl':
        lr = params.pop('lr', 0.001)
        weight_decay = params.pop('weight_decay', 0.0)
    else:
        params.pop('lr', None)
        params.pop('weight_decay', None)
    
    if args.cv:
        all_cv_games = DATASET_GAMES['train'] + DATASET_GAMES['val']
        folds = [
            {'val': all_cv_games[:2], 'train': all_cv_games[2:]},
            {'val': all_cv_games[2:4], 'train': all_cv_games[:2] + all_cv_games[4:]},
            {'val': all_cv_games[4:6], 'train': all_cv_games[:4] + all_cv_games[6:]},
            {'val': all_cv_games[6:], 'train': all_cv_games[:6]}
        ]
    else:
        all_cv_games = DATASET_GAMES['train'] + DATASET_GAMES['val']
        folds = [{'val': DATASET_GAMES['val'], 'train': DATASET_GAMES['train']}]
    
    oof_probs = []
    oof_labels = []
    oof_dfs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = 6 if args.data == 'pva' else 2
    
    for fold_idx, fold in enumerate(folds):
        X_train_full = np.concatenate([data_cache[window][g]['X'] for g in fold['train']])
        y_train_full = np.concatenate([data_cache[window][g]['y'] for g in fold['train']])
        
        X_val = np.concatenate([data_cache[window][g]['X'] for g in fold['val']])
        y_val = np.concatenate([data_cache[window][g]['y'] for g in fold['val']])
        df_val = pd.concat([data_cache[window][g]['df'] for g in fold['val']], ignore_index=True)
            
        stage_label = f"T{trial.number} F{fold_idx+1}" if args.cv else f"T{trial.number}"
        
        if model_family == 'ml':
            # Balance Train Set (2:1 Ratio just like original pipeline)
            idx_flying = np.where(y_train_full == 0)[0]
            idx_events = np.where(y_train_full != 0)[0]
            
            rng = np.random.default_rng(seed=trial.number)  # Different seed per trial for diverse sampling
            if len(idx_flying) > len(idx_events):
                idx_flying = rng.choice(idx_flying, size=len(idx_events)*2, replace=False)
                
            balanced_indices = np.concatenate([idx_flying, idx_events])
            rng.shuffle(balanced_indices)
            
            X_train = X_train_full[balanced_indices]
            y_train = y_train_full[balanced_indices]
            
            model = get_ml_model(args.model, num_classes=3, **params)
            model.fit(X_train, y_train)
            probs_val = model.predict_proba(X_val)
            
        elif model_family == 'dl':
            cw = compute_class_weight('balanced', classes=np.unique(y_train_full), y=y_train_full)
            class_weights = torch.FloatTensor(cw).to(device)
            
            train_loader = DataLoader(TrajectoryDataset(X_train_full, y_train_full, augment=True), batch_size=64, shuffle=True)
            val_loader = DataLoader(TrajectoryDataset(X_val, y_val, augment=False), batch_size=64, shuffle=False)
            
            model = get_dl_model(args.model, window, in_channels, num_classes=3).to(device)
            
            best_epoch, best_weights = train_dl_stage(
                model, train_loader, val_loader, args.epochs, device, None, stage_label, 
                class_weights=class_weights, loss_type='ce', lr=lr, weight_decay=weight_decay,
                early_stopping_metric='auc', eval_metric='f1', df_val=df_val, nms_window=args.nms
            )
            
            model.load_state_dict(best_weights)
            model.eval()
            
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    out = model(X_batch.to(device))
                    all_preds.append(out.cpu())
                    
            logits = torch.cat(all_preds, dim=0)
            probs_val = torch.softmax(logits, dim=1).numpy()
            
        oof_probs.append(probs_val)
        oof_labels.append(y_val)
        oof_dfs.append(df_val)
    
    # Calculate Global Out-Of-Fold (OOF) AUC-F1
    all_probs = np.concatenate(oof_probs)
    all_y = np.concatenate(oof_labels)
    all_df = pd.concat(oof_dfs, ignore_index=True)
    
    auc_val, peak_f1, best_thresh = get_event_macro_metric_curve_area(all_y, all_probs, all_df, args.nms, metric='f1', return_all=True)
    
    print(f"  -> Trial {trial.number} complete! Peak F1: {peak_f1:.4f} (at thresh {best_thresh:.3f}) | AUC: {auc_val:.4f}")
    
    trial.set_user_attr('peak_f1', float(peak_f1))
    trial.set_user_attr('best_thresh', float(best_thresh))
    
    return auc_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgboost', choices=ML_MODELS + DL_MODELS)
    parser.add_argument('--data', type=str, default='p', choices=['p', 'pva'])
    parser.add_argument('--nms', type=int, default=3)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30, help="Max epochs for DL models")
    parser.add_argument('--cv', action='store_true', help="Run 4-fold Cross Validation instead of just a Train/Val split")
    parser.add_argument('--restart', action='store_true', help='Delete the old tuning DB and start fresh')
    args = parser.parse_args()
    
    model_family = 'dl' if args.model in DL_MODELS else 'ml'
    
    config_path = 'configs/tune_parameters.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.model not in config:
        raise ValueError(f"No tuning config found for {args.model} in {config_path}")
        
    param_space = config[args.model]
    
    data_cache = {}
    windows_to_cache = param_space['window']['choices']
    print(f"Caching {model_family.upper()} datasets for window sizes: {windows_to_cache}...")
    
    all_games = DATASET_GAMES['train'] + DATASET_GAMES['val']
    
    for w in windows_to_cache:
        data_cache[w] = {}
        print(f"  -> Extracting features for window={w}...")
        for g in all_games:
            df_g, _, _ = load_data('data/filtered_dataset.csv', w, model_family, balance=False, train_games=[g], val_games=[], test_games=[])
            X_g = extract_features(df_g, w, model_family, args.data)
            y_g = df_g['target_multi'].values
            data_cache[w][g] = {'X': X_g, 'y': y_g, 'df': df_g}
    
    print(f"Starting Optuna sweep for {args.model} with {args.trials} trials...")
    
    study_name = f"{args.model}_tuning"
    db_path = "configs/optuna_tuning.db"
    
    if args.restart and os.path.exists(db_path):
        print("--restart flag detected. Deleting old tuning database...")
        os.remove(db_path)
        
    db_uri = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=study_name,
        storage=db_uri,
        load_if_exists=True
    )
    
    print(f"Study loaded. Currently has {len(study.trials)} finished trials.")
    study.optimize(lambda trial: objective(trial, args, data_cache, param_space, model_family), n_trials=args.trials)
    
    print("\n=== Tuning Complete ===")
    print(f"Best Val AUC: {study.best_value:.4f}")
    if 'peak_f1' in study.best_trial.user_attrs:
        print(f"Best Trial Peak F1: {study.best_trial.user_attrs['peak_f1']:.4f}")
        print(f"Best Trial Optimal Threshold: {study.best_trial.user_attrs['best_thresh']:.3f}")
    
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save top 3 best parameters
    os.makedirs('configs', exist_ok=True)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed_trials[:3]
    
    print("\n--- TOP 3 CONFIGURATIONS ---")
    for idx, t in enumerate(top_trials):
        rank = idx + 1
        print(f"Rank {rank}: Trial {t.number} (AUC: {t.value:.4f} | Peak F1: {t.user_attrs.get('peak_f1', 0):.4f})")
        out_path = f"configs/best_{args.model}_top{rank}.yaml"
        with open(out_path, 'w') as f:
            yaml.dump(t.params, f)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
