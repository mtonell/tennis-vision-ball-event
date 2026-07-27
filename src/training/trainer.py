import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np

from src.utils.metrics import apply_nms_and_tolerance, get_event_macro_metric, get_event_macro_metric_curve_area

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_dl_stage(model, train_loader, val_loader, epochs, device, experiment, stage_name, class_weights=None, loss_type='ce', lr=0.001, weight_decay=0.0, early_stopping_metric='loss', eval_metric='f1', df_val=None, nms_window=3):
    if loss_type == 'focal':
        criterion = FocalLoss(weight=class_weights.to(device) if class_weights is not None else None)
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_metric_val = float('inf') if early_stopping_metric == 'loss' else -1.0
    best_weights = None
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        val_loss, correct, total, val_acc = 0, 0, 0, 0
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            all_preds_logits = []
            all_y = []
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    val_loss += criterion(out, y).item()
                    
                    if early_stopping_metric in ['auc', 'peak']:
                        all_preds_logits.append(out.cpu())
                        all_y.append(y.cpu())
                    else:
                        preds = torch.argmax(out, dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                        
            val_loss = val_loss / len(val_loader)
            
            if early_stopping_metric in ['auc', 'peak']:
                logits = torch.cat(all_preds_logits, dim=0)
                y_true = torch.cat(all_y, dim=0).numpy()
                probs = torch.softmax(logits, dim=1).numpy()
                max_probs = np.max(probs, axis=1)
                
                if early_stopping_metric == 'auc':
                    if df_val is not None:
                        current_metric = get_event_macro_metric_curve_area(y_true, probs, df_val, nms_window, metric=eval_metric)
                    else:
                        current_metric = 0 
                    print(f"[{stage_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val AUC-{eval_metric.upper()}: {current_metric:.4f}")
                
                elif early_stopping_metric == 'peak':
                    best_score = -1
                    for thresh in np.linspace(0.1, 0.9, 9):
                        preds = np.argmax(probs, axis=1)
                        preds[(preds != 0) & (max_probs < thresh)] = 0
                        if df_val is not None:
                            preds = apply_nms_and_tolerance(y_true, preds, max_probs, df_val, nms_window)
                        score = get_event_macro_metric(y_true, preds, metric=eval_metric)
                        if score > best_score:
                            best_score = score
                    current_metric = best_score
                    print(f"[{stage_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Peak-{eval_metric.upper()}: {current_metric:.4f}")
                    
                is_best = current_metric > best_metric_val
            else:
                val_acc = correct / total if total > 0 else 0
                current_metric = val_loss
                print(f"[{stage_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                is_best = current_metric < best_metric_val
            
            if is_best:
                best_metric_val = current_metric
                best_epoch = epoch + 1
                patience_counter = 0
                best_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                
            if experiment:
                metrics_to_log = {
                    f'{stage_name}_train_loss': train_loss/len(train_loader), 
                    f'{stage_name}_val_loss': val_loss,
                }
                if early_stopping_metric == 'auc':
                    metrics_to_log[f'{stage_name}_val_auc_{eval_metric}'] = current_metric
                elif early_stopping_metric == 'peak':
                    metrics_to_log[f'{stage_name}_val_peak_{eval_metric}'] = current_metric
                else:
                    metrics_to_log[f'{stage_name}_val_acc'] = val_acc
                experiment.log_metrics(metrics_to_log, step=epoch)
                
            if patience_counter >= patience:
                print(f"[{stage_name}] Early stopping triggered at epoch {epoch+1}!")
                break
        else:
            print(f"[{stage_name}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | (Final Validation Model - No Val)")
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            if experiment:
                experiment.log_metrics({f'{stage_name}_train_loss': train_loss/len(train_loader)}, step=epoch)
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return best_epoch, best_weights




def train_ml_stage(model, X_train, y_train, stage_name="ML Stage"):
    """Wraps ML training to maintain consistent API with DL models."""
    print(f"[{stage_name}] Fitting ML model...")
    model.fit(X_train, y_train)
    return model
