import os
import pandas as pd
import argparse

def filter_dataset(input_dir, output_file):
    print(f"Scanning '{input_dir}' for Label.csv files...")
    all_dfs = []
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    games = [g for g in os.listdir(input_dir) if g.startswith('game')]
    for game in games:
        game_path = os.path.join(input_dir, game)
        if not os.path.isdir(game_path):
            continue
            
        clips = os.listdir(game_path)
        for clip in clips:
            clip_path = os.path.join(game_path, clip)
            if not os.path.isdir(clip_path):
                continue
                
            csv_path = os.path.join(clip_path, 'Label.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['game'] = game
                df['clip'] = clip
                all_dfs.append(df)
                
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        master_df.to_csv(output_file, index=False)
        
        total_clips = len(all_dfs)
        total_frames = len(master_df)
        avg_frames_per_clip = total_frames / total_clips if total_clips > 0 else 0
         
        valid_status = master_df['status'].dropna().astype(int)
        class_counts = valid_status.value_counts().to_dict()
        flying = class_counts.get(0, 0)
        hit = class_counts.get(1, 0)
        bounce = class_counts.get(2, 0)
        
        print("\n--- Dataset Aggregation Complete ---")
        print(f"Saved dataset to: {output_file}")
        print("\nDataset Statistics:")
        print(f"- Total clips found: {total_clips}")
        print(f"- Total ball positions (frames): {total_frames}")
        print(f"- Average positions per clip: {avg_frames_per_clip:.1f}")
        print("\nClass Imbalance:")
        print(f"- Flying frames (0): {flying}")
        print(f"- Hit frames (1):    {hit}")
        print(f"- Bounce frames (2): {bounce}")
        print("------------------------------------")
    else:
        print("No Label.csv files were found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="Dataset", help="Path to the original Dataset directory.")
    parser.add_argument("--output_file", default=os.path.join("data", "filtered_dataset.csv"), help="Path for the aggregated output CSV.")
    args = parser.parse_args()
    
    filter_dataset(args.input_dir, args.output_file)
