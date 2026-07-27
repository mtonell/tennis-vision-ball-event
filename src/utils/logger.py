import os
from dotenv import load_dotenv
from src.utils.config import TRACKING, COMET_PROJECT

load_dotenv()
try:
    from comet_ml import Experiment
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

class ExperimentLogger:
    def __init__(self, args, model_family):
        self.experiment = None
        
        comet_key = os.getenv('COMET_API_KEY')
        track = TRACKING
        
        # If true and no env, put it false
        if track and not comet_key:
            track = False
            
        comet_project = COMET_PROJECT
        
        if track and comet_key and HAS_COMET:
            self.experiment = Experiment(project_name=comet_project, api_key=comet_key)
            
            if getattr(args, 'run_name', None):
                run_name = args.run_name
            else:
                loss_suffix = '_focal' if getattr(args, 'loss', 'ce') == 'focal' and model_family == 'dl' else ''
                data_suffix = '_pva' if args.data == 'pva' and model_family == 'dl' else ''
                cv_suffix = '_cv' if getattr(args, 'cv', False) else ''
                run_name = f"{args.model}_stage{getattr(args, 'stages', 1)}_win{args.window}{loss_suffix}{data_suffix}{cv_suffix}"
                
            self.experiment.set_name(run_name)
            self.experiment.log_parameters(vars(args))
            print(f"Logging to CometML. Run name: {run_name}")
        else:
            print("CometML tracking is disabled.")

    def log_metric(self, name, value, step=None):
        if self.experiment:
            self.experiment.log_metric(name, value, step=step)
            
    def log_metrics(self, metrics_dict, step=None):
        if self.experiment:
            self.experiment.log_metrics(metrics_dict, step=step)

    def log_curve(self, name, x, y):
        if self.experiment:
            self.experiment.log_curve(name, x, y)
            
    def log_confusion_matrix(self, y_true, y_pred):
        if self.experiment:
            self.experiment.log_confusion_matrix(y_true, y_pred)
            
    def end(self):
        if self.experiment:
            self.experiment.end()
