import os
import pandas as pd
from dotenv import load_dotenv
from src.utils.config import COMET_PROJECT

try:
    from comet_ml.api import API
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

load_dotenv()

def print_comet_ranking():
    if not HAS_COMET:
        print("comet_ml is not installed. Run: pip install comet-ml")
        return
    
    comet_key = os.getenv('COMET_API_KEY')
    if not comet_key:
        print("COMET_API_KEY is not set in your .env file. Cannot fetch rankings.")
        return
    
    try:
        api = API()
    except Exception as e:
        print(f"Failed to connect to CometML: {e}")
        return
    
    project_name = COMET_PROJECT
    
    workspaces = api.get_workspaces()
    if not workspaces:
        print("No Comet workspaces found for this API key.")
        return
        
    workspace = workspaces[0]
    
    try:
        experiments = api.get_experiments(workspace, project_name)
    except Exception as e:
        print(f"Could not fetch experiments: {e}")
        return
        
    if not experiments:
        print(f"No experiments found in project {workspace}/{project_name}")
        return
        
    results = []
    for exp in experiments:
        metrics = exp.get_metrics_summary()
        metrics_dict = {m['name']: float(m['valueCurrent']) for m in metrics if m['valueCurrent'] is not None}
        
        results.append({
            'Run Name': exp.name,
            'CV-Tuned F1': metrics_dict.get('test_event_macro_f1_tuned', 0.0),
            'Baseline F1': metrics_dict.get('test_event_macro_f1_baseline', 0.0),
            'Oracle Max F1': metrics_dict.get('test_event_macro_f1_oracle', 0.0),
            'FPS': metrics_dict.get('inference_fps', 0.0),
            'Speed (ms/frame)': metrics_dict.get('inference_speed_ms', 0.0)
        })
        
    df = pd.DataFrame(results)
    
    df = df[df['CV-Tuned F1'] > 0]
    
    df_sorted = df.sort_values(by='CV-Tuned F1', ascending=False).reset_index(drop=True)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    
    print("\n" + "="*90)
    print("COMET.ML EXPERIMENT RANKING")
    print("="*90)
    print(df_sorted.to_string())
    print("="*90 + "\n")

if __name__ == '__main__':
    print_comet_ranking()
