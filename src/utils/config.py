import argparse

DL_MODELS = ['cnn', 'lstm', 'bilstm', 'tcn', 'transformer', 'resnet', 'inception']
ML_MODELS = ['catboost', 'rf', 'xgboost', 'lgbm']

TRACKING = False
COMET_PROJECT = "tennis-vision-ball-event"

DATASET_GAMES = {
    'train': ["game1", "game2", "game3", "game4", "game5", "game6"],
    'val': ["game7", "game8"],
    'test': ["game9", "game10"]
}

def get_parser(is_orchestrator=False):
    parser = argparse.ArgumentParser()
    if is_orchestrator:
        parser.add_argument('--model', type=str, nargs='+', default=['all'], help="Models to run (e.g. 'all' or 'cnn resnet')")
        parser.add_argument('--window', type=str, nargs='+', default=['all'], help="Window sizes (e.g. 'all' or '5 9')")
        parser.add_argument('--stages', type=str, nargs='+', default=['1'], help="Stages to run (e.g. 'all' or '1 2')")
        parser.add_argument('--loss', type=str, nargs='+', default=['ce'], help="Loss functions (e.g. 'all' or 'ce focal')")
        parser.add_argument('--nms', type=int, default=3, help="NMS window size (default: 3)")
    else:
        parser.add_argument('--model', type=str, choices=DL_MODELS + ML_MODELS, default='cnn')
        parser.add_argument('--window', type=int, default=9, help="Must be an odd number (e.g. 3, 5, 7)")
        parser.add_argument('--stages', type=int, choices=[1, 2, 3], default=1)
        parser.add_argument('--loss', type=str, choices=['ce', 'focal'], default='ce', help='Loss function for DL models')
        parser.add_argument('--nms', type=int, default=0, help="Window size for NMS (e.g., 3)")
        
    parser.add_argument('--data', type=str, choices=['p', 'pva'], default='p', help='Features to use')
    parser.add_argument('--dataset_path', type=str, default='data/filtered_dataset.csv')
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--run_name', type=str, default=None, help="Custom name for Comet ML log")
    parser.add_argument('--save', action='store_true', help='Save the trained model weights')
    parser.add_argument('--deploy', action='store_true', help='Train on all data for deployment')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--cv', action='store_true', help='Run 4-Fold Cross Validation')
    parser.add_argument('--metric', type=str, default='loss', choices=['loss', 'auc', 'peak'], help='Early stopping metric')
    parser.add_argument('--parameters', type=str, default=None, help="Path to YAML file with hyperparameters")
    
    return parser
