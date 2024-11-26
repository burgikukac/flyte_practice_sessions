from flytekit import task, workflow, map_task
from ray import tune
import numpy as np

# Black-box model function to evaluate each configuration
@task
def evaluate_model(x: float, y: float) -> float:
    # Custom loss function (replace with actual model loss calculation)
    loss = (x - 3) ** 2 + (y + 2) ** 2 + np.random.normal(0, 0.1)
    return loss

# Flyte workflow to execute Ray Tune with Flyte as the backend
@workflow
def hyperopt_with_flyte_workflow(num_samples: int = 50, max_parallel_trials: int = 5) -> dict:
    # Set up Ray Tune's search space (without running a Ray cluster)
    search_space = {
        "x": tune.uniform(-5, 5),
        "y": tune.uniform(-5, 5)
    }

    # Use Ray Tune to suggest configurations for each trial
    search_algo = tune.search.hyperopt.HyperOptSearch()
    trial_configs = [
        search_algo.suggest(f"trial_{i}", search_space) for i in range(num_samples)
    ]
    
    # Evaluate trials in parallel, limited by max_parallel_trials
    losses = map_task(evaluate_model, metadata={"parallelism": max_parallel_trials})(
        x=[config["x"] for config in trial_configs],
        y=[config["y"] for config in trial_configs]
    )

    # Find the best configuration based on minimum loss
    min_loss_index = np.argmin(losses)
    best_config = {
        "x": trial_configs[min_loss_index]["x"],
        "y": trial_configs[min_loss_index]["y"],
        "loss": losses[min_loss_index]
    }
    return best_config