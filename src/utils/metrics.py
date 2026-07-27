import numpy as np

def get_event_macro_metric(y_true, y_pred, metric='f1'):
    from sklearn.metrics import f1_score, fbeta_score
    labels = [1, 2] if 2 in np.unique(y_true) else [1]
    
    if metric == 'f1':
        scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    elif metric == 'f2':
        scores = fbeta_score(y_true, y_pred, labels=labels, beta=2.0, average=None, zero_division=0)
    else:
        scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return scores.mean()

def get_event_macro_metric_curve_area(y_true, probs, df, nms_window, metric='f1', return_peak=False, return_all=False):
    from sklearn.metrics import auc
    thresholds = np.linspace(0.01, 0.99, 50)
    scores = []
    
    for thresh in thresholds:
        preds = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)
        preds[(preds != 0) & (max_probs < thresh)] = 0
        
        preds = apply_nms_and_tolerance(y_true, preds, max_probs, df, nms_window)
        scores.append(get_event_macro_metric(y_true, preds, metric=metric))
        
    if return_all:
        auc_val = auc(thresholds, scores)
        peak_idx = np.argmax(scores)
        return auc_val, scores[peak_idx], thresholds[peak_idx]
        
    if return_peak:
        return max(scores)
        
    # Calculate actual Area Under the Curve 
    return auc(thresholds, scores)


def apply_nms_and_tolerance(y_true, y_pred, probs, df_test, nms_window):
    tolerance = nms_window // 2
    y_pred_final = y_pred.copy()
    df_test_temp = df_test.copy()
    df_test_temp['idx'] = np.arange(len(df_test_temp))
    for (game, clip), group in df_test_temp.groupby(['game', 'clip']):
        indices = group['idx'].values
        clip_y_pred = y_pred_final[indices]
        clip_probs = probs[indices]
        
        # NMS
        for i in range(len(indices)):
            pred = clip_y_pred[i]
            if pred != 0:
                start = max(0, i - tolerance)
                end = min(len(indices), i + tolerance + 1)
                neighborhood_preds = clip_y_pred[start:end]
                matching_indices = start + np.where(neighborhood_preds == pred)[0]
                if len(matching_indices) > 1:
                    best_idx = matching_indices[np.argmax(clip_probs[matching_indices])]
                    for idx in matching_indices:
                        if idx != best_idx:
                            clip_y_pred[idx] = 0
                            
        # Temporal Tolerance (Only if ground truth is available)
        if y_true is not None:
            clip_y_true = y_true[indices]
            for i in range(len(indices)):
                true_label = clip_y_true[i]
                if true_label != 0:
                    start = max(0, i - tolerance)
                    end = min(len(indices), i + tolerance + 1)
                    if true_label in clip_y_pred[start:end]:
                        pred_idx = start + np.where(clip_y_pred[start:end] == true_label)[0][0]
                        if pred_idx != i:
                            clip_y_pred[pred_idx] = 0
                            clip_y_pred[i] = true_label
                        
        y_pred_final[indices] = clip_y_pred
        
    return y_pred_final

def evaluate_and_log_results(y_true, preds_dict, inference_time_ms=None, experiment=None):
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    print("\n" + "="*60)
    print("                        TEST RESULTS")
    print("="*60)
    
    if inference_time_ms:
        fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0
        print(f"\nInference Speed: {inference_time_ms:.4f} ms/frame ({fps:.2f} FPS)")
        
    primary_key = 'Tuned' if 'Tuned' in preds_dict else list(preds_dict.keys())[0]
    primary_preds = preds_dict[primary_key]
    
    print(f"\n--- Detailed Classification Report ({primary_key}) ---")
    print(classification_report(y_true, primary_preds, target_names=['Flying (0)', 'Hit (1)', 'Bounce (2)'], zero_division=0))
    
    print("-"*60)
    print("Final Metrics Summary")
    print("-"*60)
    
    for key, preds in preds_dict.items():
        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        
        report = classification_report(y_true, preds, labels=[0, 1, 2], output_dict=True, zero_division=0)
        event_f1 = (report['1']['f1-score'] + report['2']['f1-score']) / 2.0
        
        suffix = " [Theoretical Max]" if key == 'Oracle' else ""
        print(f"  {key:<8} : Event Macro F1 = {event_f1:.4f} | Accuracy = {acc:.4f}{suffix}")
        
        if experiment:
            key_lower = key.lower()
            experiment.log_metric(f"test_event_macro_f1_{key_lower}", event_f1)
            experiment.log_metric(f"test_macro_f1_{key_lower}", macro_f1)
            
            if key_lower != 'tuned':
                experiment.log_metric(f"test_accuracy_{key_lower}", acc)
                
            if key == primary_key:
                experiment.log_confusion_matrix(y_true, preds)
                
    if experiment and inference_time_ms:
        experiment.log_metric("inference_speed_ms", inference_time_ms)
        experiment.log_metric("inference_fps", fps)
        
    print("="*60 + "\n")
