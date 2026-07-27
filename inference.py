import os
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import torch

from src.data.dataset import process_clip, extract_features
from src.models.dl import get_dl_model
from src.utils.metrics import apply_nms_and_tolerance
from src.utils.config import DL_MODELS, DATASET_GAMES

def run_inference(args):
    if args.input:
        print(f"Loading input file: {args.input}")
        try:
            df = pd.read_csv(args.input)
        except Exception as e:
            print(f"Error reading {args.input}: {e}")
            return
            
        if 'game' not in df.columns:
            df['game'] = 'inference_game'
        if 'clip' not in df.columns:
            df['clip'] = 'inference_clip'
            
        processed_dfs = []
        for (game, clip), clip_df in df.groupby(['game', 'clip']):
            processed = process_clip(clip_df, args.window // 2, args.model_type, args.window, is_inference=True)
            if processed is not None:
                processed_dfs.append(processed)
                
        if not processed_dfs:
            print("Error: Input data is too short or invalid after processing.")
            return
            
        df_processed = pd.concat(processed_dfs, ignore_index=True)
    else:
        print("No input provided. Falling back to test set from data/filtered_dataset.csv...")
        if not os.path.exists('data/filtered_dataset.csv'):
            print("No test set found at data/filtered_dataset.csv. Exiting.")
            return
            
        from src.data.dataset import load_data
        test_g = DATASET_GAMES['test']
        
        _, _, df_processed = load_data('data/filtered_dataset.csv', args.window, args.model_type, False, [], [], test_g)
        
        if len(df_processed) == 0:
            print("No test data found.")
            return
            
    print(f"Processing {len(df_processed)} frames...")
    
    X = extract_features(df_processed, args.window, args.model_type, data_type=args.data)
    
    print(f"Loading weights from {args.weights}")
    if args.model_type == 'dl':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_channels = 6 if args.data == 'pva' else 2
        model = get_dl_model(args.model, args.window, in_channels, 3).to(device)
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        model.eval()
        
        start_time = time.time()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        inference_time_ms = (time.time() - start_time) * 1000 / len(X)
    else:
        model = joblib.load(args.weights)
        start_time = time.time()
        probs = model.predict_proba(X)
        inference_time_ms = (time.time() - start_time) * 1000 / len(X)
        
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    
    if args.nms > 0:
        print(f"Applying Non-Maximum Suppression (window={args.nms})")
        preds = apply_nms_and_tolerance(None, preds, max_probs, df_processed, args.nms)
        max_probs = np.where(preds != 0, max_probs, 0.0)
        
    df_processed['predicted_status'] = preds
    df_processed['predicted_prob'] = max_probs
    
    fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0
    print(f"Inference Speed: {inference_time_ms:.4f} ms/frame ({fps:.2f} FPS)")
    
    if args.output:
        df_processed.to_csv(args.output, index=False)
        print(f"Full predictions saved to {args.output}")
        
    events = df_processed[df_processed['predicted_status'] != 0]
    if len(events) == 0:
        print("No hits or bounces detected.")
    else:
        print("\n=== Detected Events ===")
        # Sort by game, clip, and index to print chronologically
        for (game, clip), group in events.groupby(['game', 'clip']):
            print(f"\nGame: {game} | Clip: {clip}")
            for idx, row in group.iterrows():
                event_type = 'Hit' if row['predicted_status'] == 1 else 'Bounce'
                frame_val = row.get('frame_num', None)
                frame = int(frame_val) if frame_val is not None and pd.notna(frame_val) else idx
                print(f"  Frame {frame:<5} -> {event_type:<6} (Prob: {row['predicted_prob']:.2f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on tennis ball tracking data")
    parser.add_argument('--input', type=str, default=None, help="Path to input CSV file. If not provided, runs on test set.")
    parser.add_argument('--output', type=str, default=None, help="Path to save output CSV file with predictions.")
    parser.add_argument('--model', type=str, default='resnet', help="Model architecture name (e.g. resnet, xgboost)")
    parser.add_argument('--weights', type=str, default='weights/resnet_stage1_win9.pth', help="Path to trained model weights")
    parser.add_argument('--window', type=int, default=9, help="Window size used during training")
    parser.add_argument('--data', type=str, default='p', help="Features used (p or pva)")
    parser.add_argument('--nms', type=int, default=3, help="NMS window size")
    
    args = parser.parse_args()
    
    args.model_type = 'dl' if args.model in DL_MODELS else 'ml'
    
    run_inference(args)
