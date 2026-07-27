import itertools
from src.utils.config import get_parser, DL_MODELS, ML_MODELS
import train

def main():
    parser = get_parser(is_orchestrator=True)
    args = parser.parse_args()



    ALL_WINDOWS = [5, 9, 13]
    ALL_STAGES = [1, 2, 3]
    ALL_LOSSES = ['ce', 'focal']

    if 'all' in args.model:
        models_dl = DL_MODELS
        models_ml = ML_MODELS
    else:
        models_dl = [m for m in args.model if m in DL_MODELS]
        models_ml = [m for m in args.model if m in ML_MODELS]

    parameters = getattr(args, 'parameters', None)
    if 'all' in args.window:
        windows = ALL_WINDOWS
    else:
        windows = [int(w) for w in args.window]

    if parameters:
        import yaml
        with open(parameters, 'r') as f:
            custom = yaml.safe_load(f) or {}
        if 'window' in custom:
            windows = [custom['window']]
            print(f"Overriding window sizes with {windows[0]} from {parameters}")

    if 'all' in args.stages:
        stages = ALL_STAGES
    else:
        stages = [int(s) for s in args.stages]

    if 'all' in args.loss:
        losses = ALL_LOSSES
    else:
        losses = args.loss

    nms = args.nms
    
    print("STARTING TENNIS VISION GRID SEARCH")
    
    if models_dl:
        for model, window, stage, loss in itertools.product(models_dl, windows, stages, losses):
            print(f"\n{'-'*60}")
            print(f"RUNNING: Model={model.upper()} | Window={window} | Stages={stage} | Loss={loss.upper()}")
            print(f"{'-'*60}")
            
            cmd_args = ["--model", model, "--stages", str(stage), "--window", str(window), "--nms", str(nms), "--loss", loss, "--data", args.data, "--metric", args.metric]
            if getattr(args, 'cv', False): cmd_args.append("--cv")
            if getattr(args, 'deploy', False): cmd_args.append("--deploy")
            if getattr(args, 'save', False): cmd_args.append("--save")
            if parameters: cmd_args.extend(["--parameters", parameters])
            
            try: 
                train_parser = get_parser(is_orchestrator=False)
                run_args = train_parser.parse_args(cmd_args)
                train.run_experiment(run_args)
            except Exception as e: 
                print(f"Error occurred while running {model} (Win: {window}, Stage: {stage}, Loss: {loss}). Error: {e}. Skipping...")

   
    if models_ml:
        for model, window, stage in itertools.product(models_ml, windows, stages):
            print(f"\n{'-'*60}")
            print(f"RUNNING: Model={model.upper()} | Window={window} | Stages={stage}")
            print(f"{'-'*60}")
            
            cmd_args = ["--model", model, "--stages", str(stage), "--window", str(window), "--nms", str(nms), "--data", args.data, "--metric", args.metric]
            if getattr(args, 'cv', False): cmd_args.append("--cv")
            if getattr(args, 'deploy', False): cmd_args.append("--deploy")
            if getattr(args, 'save', False): cmd_args.append("--save")
            if parameters: cmd_args.extend(["--parameters", parameters])
            
            try: 
                train_parser = get_parser(is_orchestrator=False)
                run_args = train_parser.parse_args(cmd_args)
                train.run_experiment(run_args)
            except Exception as e: 
                print(f"Error occurred while running {model} (Win: {window}, Stage: {stage}). Error: {e}. Skipping...")

    print("\nALL EXPERIMENTS COMPLETED SUCCESSFULLY")

if __name__ == '__main__':
    main()
