from prefect import flow
from src.tasks.retraining_tasks import (
    check_drift_and_trigger,
    collect_latest_data,
    train_candidate_model,
    evaluate_candidate,
    register_and_deploy
)

@flow(name="automated_retraining_flow")
def automated_retraining_flow(force_retrain: bool = False):
    """
    Orchestrate the retraining pipeline.
    """
    should_train = check_drift_and_trigger()
    
    if force_retrain or should_train:
        data = collect_latest_data()
        
        # Params from config or best hparams
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1
        }
        
        run_id = train_candidate_model(data, params)
        
        current_prod_uri = "models:/heart-disease-model/Production"
        is_better = evaluate_candidate(run_id, current_prod_uri)
        
        if is_better:
            register_and_deploy(run_id)
