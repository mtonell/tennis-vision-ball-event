import os
try:
    import comet_ml
except ImportError:
    pass
import torch

from src.utils.config import get_parser, DL_MODELS, DATASET_GAMES
from src.utils.logger import ExperimentLogger
from src.data.dataset import load_data, extract_features
from src.pipelines.single_stage import run_single_stage_pipeline
from src.pipelines.multi_stage import run_multi_stage_pipeline
from src.utils.seed import set_seed

def run_experiment(args):
    set_seed(42)
    
    if args.window % 2 == 0:
        raise ValueError("Window must be odd")
        
    if args.parameters:
        import yaml
        import src.models.ml as ml_module
        with open(args.parameters, 'r') as f:
            custom_params = yaml.safe_load(f)
            if custom_params:
                if 'window' in custom_params:
                    args.window = custom_params.pop('window')
                    print(f"Overriding window size to {args.window} based on {args.parameters}")
                if 'lr' in custom_params:
                    args.lr = custom_params.pop('lr')
                    print(f"Overriding learning_rate to {args.lr}")
                if 'weight_decay' in custom_params:
                    args.weight_decay = custom_params.pop('weight_decay')
                    print(f"Overriding weight_decay to {args.weight_decay}")
                
                filename = os.path.basename(args.parameters).replace('.yaml', '')
                cv_suffix = '_cv' if args.cv else ''
                args.run_name = f"{filename}{cv_suffix}"
                
                args.custom_model_params = custom_params
                ml_module.GLOBAL_KWARGS = custom_params
                print(f"Injected custom hyperparameters from {args.parameters}")
        
    model_family = 'dl' if args.model in DL_MODELS else 'ml'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = ExperimentLogger(args, model_family)
        
    # Dataset splits
    if args.deploy:
        train_g = DATASET_GAMES['train'] + DATASET_GAMES['val'] + DATASET_GAMES['test']
        val_g = []
        test_g = []
        print("\nDEPLOY MODE ENABLED: Combining all games into training set! No validation will occur.\n")
    else:
        train_g = DATASET_GAMES['train']
        val_g = DATASET_GAMES['val']
        test_g = DATASET_GAMES['test']
    
    # Cross-Validation or Standard Benchmarking
    if args.cv:
        from src.pipelines.cv import run_cv_pipeline
        run_cv_pipeline(args, device, experiment, args.dataset_path, args.nms, model_family)
        return
    
    print(f"Loading data... Model Family: {model_family}, Window: {args.window}")
    # Focal Loss is mathematically designed to be trained on the RAW imbalanced dataset.
    # If we use Focal Loss, we explicitly disable dataset balancing
    should_balance = False if args.loss == 'focal' else True
    
    df_train, df_val, df_test = load_data(
        args.dataset_path, args.window, model_family, balance=should_balance,
        train_games=train_g, val_games=val_g, test_games=test_g
    )
    
    print(f"Extracting Features... Data Type: {args.data}")
    X_train = extract_features(df_train, args.window, model_family, data_type=args.data)
    X_val = extract_features(df_val, args.window, model_family, data_type=args.data)
    X_test = extract_features(df_test, args.window, model_family, data_type=args.data)
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    if args.stages >= 2:
        run_multi_stage_pipeline(args, df_train, df_val, df_test, X_train, X_val, X_test, model_family, device, experiment)
    else:
        run_single_stage_pipeline(args, df_train, df_val, df_test, X_train, X_val, X_test, model_family, device, experiment, getattr(getattr(experiment, 'experiment', None), 'name', None) or f"{args.model}_stage1")
        
    if experiment:
        experiment.end()

def main():
    parser = get_parser(is_orchestrator=False)
    args = parser.parse_args()
    run_experiment(args)

if __name__ == '__main__':
    main()
