from flytekit import task, workflow, map_task, dynamic
from flytekit.types.file import FlyteFile
from ray import tune
import xgboost as xgb
import pandas as pd
import numpy as np
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.hyperopt import HyperOptSearch
import polars as pl
# Task to load the Parquet file and return a DataFrame
@task
def load_data(parquet_path: str) -> pd.DataFrame:
    data = pd.read_parquet(parquet_path)
    return data

# Task to split the data into training and validation sets
@task
def split_data(
    traintest: pd.DataFrame,
    date_col: str,
    train_length: int = 270,
    valid_length: int = 30,
    step_size: int = 30,
    n_splits: int = 1,
    current_split: int = 1,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataset into training and validation sets based on specified parameters.

    Args:
        traintest (pd.DataFrame): The full dataset containing a date column.
        date_col (str): The name of the date column in the dataset.
        train_length (int): The number of days to include in the training set. Default is 270.
        valid_length (int): The number of days to include in the validation set. Default is 30.
        step_size (int): The number of days to shift the train-validation split window. Default is 30.
        n_splits (int): The total number of splits to consider. Default is 1.
        current_split (int): The current split index (1-based). Default is 1.

    Returns:
        pd.DataFrame: Training data subset.
        pd.DataFrame: Validation data subset.
    """
    # Calculate the last day of the training set
    last_train_day = (
        traintest[date_col].max() - valid_length
        - (n_splits * step_size)
        + (current_split * step_size)
    )
    
    # Calculate the first day of the training set
    first_train_day = last_train_day - train_length + 1
    
    # Filter the training data
    train_data = traintest[
        (traintest[date_col] >= first_train_day) & (traintest[date_col] <= last_train_day)
    ]
    
    # Calculate the validation set range
    first_valid_day = last_train_day + 1
    last_valid_day = last_train_day + valid_length
    
    # Filter the validation data
    valid_data = traintest[
        (traintest[date_col] > last_train_day) & (traintest[date_col] <= last_valid_day)
    ]
    
    return train_data, valid_data


# Task to train the XGBoost model and return the validation loss
@task
def train_xgboost(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    n_trees: int,
    max_depth: int,
    target_column: str = "price_m2"
) -> float:
    cols_to_keep =       [
          "med_price_m2",
          "condition_type_id",
          "year_built",
          "zone_id",
          "mean_area_size",
          "balcony_area_ratio",
          "lon",
          "diff_condition_type_id",
          "diff_area_size",
          "area_size",
          "lat",
          "pred_date_int",
          "heating_type_id",
          "location_id",
          "comfort_level_id",
          "street_id",
          "mean_room_count",
          "mean_condition_type_id",
          "floor_id",
          "property_subtype_id",
          "view_type_id",
          "avg_room_size",
          "elevator_type_id",
          "diff_room_count",
        ]
    cols_to_keep.append(target_column)
    cols_to_keep.append("pred_date_int")
    train_data = train_data[cols_to_keep]
    val_data = val_data[cols_to_keep]
    dtrain = xgb.DMatrix(data = train_data.drop(columns=[target_column]).to_numpy(), label=train_data[target_column].values)
    dval = xgb.DMatrix(data = val_data.drop(columns=[target_column]).to_numpy(), label=val_data[target_column].values)

    params = {
        "objective" : "reg:squarederror",
        "nthread": 4,
        "max_depth": max_depth
    }
    model = xgb.train(params, dtrain, num_boost_round=n_trees, evals=[(dval, "validation")], verbose_eval=False)
    val_loss = model.eval(dval)
    return float(val_loss.split(":")[1]) #{'loss' :float(val_loss.split(":")[1])}

# Workflow to perform hyperparameter tuning
@dynamic
def hyperopt_with_xgboost_workflow(
    parquet_path: str,
    num_samples: int = 10,
    max_parallel_trials: int = 3
) -> dict:
    data = load_data(parquet_path=parquet_path)
    train_data, val_data = split_data(traintest=data, date_col="pred_date_int")

    # Define the search space
    search_space = {
        "n_trees": tune.randint(50, 200),
        "max_depth": tune.randint(3, 10)
    }

    # Use Ray Tune to suggest configurations for each trial
    #search_algo = HyperOptSearch()
    search_algo = OptunaSearch(space = search_space, metric = 'loss', mode = 'min')
    trial_configs = [
        search_algo.suggest(f"trial_{i}") for i in range(num_samples)
    ]

    # Evaluate each trial in parallel
    # 
    
    losses = train_xgboost(train_data=train_data, val_data=val_data, n_trees=100, max_depth=12)
    # Find the best configuration
    min_loss_index = np.argmin(losses)
    best_config = {
        "n_trees": trial_configs[min_loss_index]["n_trees"],
        "max_depth": trial_configs[min_loss_index]["max_depth"],
        "loss": losses[min_loss_index]
    }
    return best_config

@task
def print_df_stats(df: pl.DataFrame):
    print(df.describe())
    print(df.schema)

@task
def load_polars_data(input_file: FlyteFile) -> pl.DataFrame:
    input_file.download()
    data = pl.read_parquet(input_file.path)
    return data

@task
def train_xgboost_polars(
    traintest: pl.DataFrame,
#    val_data: pl.DataFrame,
    n_trees: int,
    max_depth: int,
    target_column: str = "price_m2", 
    date_col: str = "pred_date_int",
) -> float:
    cols_to_keep =       [
          "med_price_m2",
          "condition_type_id",
          "year_built",
          "zone_id",
          "mean_area_size",
          "balcony_area_ratio",
          "lon",
          "diff_condition_type_id",
          "diff_area_size",
          "area_size",
          "lat",
          "pred_date_int",
          "heating_type_id",
          "location_id",
          "comfort_level_id",
          "street_id",
          "mean_room_count",
          "mean_condition_type_id",
          "floor_id",
          "property_subtype_id",
          "view_type_id",
          "avg_room_size",
          "elevator_type_id",
          "diff_room_count",
        ]
    if target_column not in cols_to_keep:
        cols_to_keep.append(target_column)
    if date_col not in cols_to_keep:
        cols_to_keep.append(date_col)
    traintest = traintest.select(cols_to_keep)

    valid_length = 30
    step_size = 30
    n_splits = 1
    current_split = 1
    train_length = 270

    traintest = traintest.to_pandas()


    last_train_day = (
        (traintest[date_col].max() - valid_length)
        - (n_splits * step_size)
        + (current_split * step_size)
    )
    first_train_day = last_train_day - train_length + 1
    train_data = traintest[
        (traintest[date_col] >= first_train_day) & (traintest[date_col] <= last_train_day)
    ]
    first_valid_day = last_train_day + 1
    last_valid_day = last_train_day + valid_length

    valid_data = traintest[
        (traintest[date_col] > last_train_day)
        & (traintest[date_col] <= last_train_day + valid_length)
    ]
    price_weight = (
                (1 / train_data[target_column]).abs().clip(upper=100).to_numpy()
            )  # MAPE objective: MAE (pseudohuber, logcosh) with 1/price weights
            # Train the model (date column is retained)
    train_matrix = xgb.DMatrix(
                data=train_data.drop(columns=[target_column]), label=train_data[target_column], weight=price_weight
            )
    valid_matrix = xgb.DMatrix(data=valid_data.drop(columns=[target_column]), label=valid_data[target_column])
    #dtrain = xgb.DMatrix(data = train_data.drop(columns=[target_column]), label=train_data[target_column])
    #dval = xgb.DMatrix(data = valid_data.drop(columns=[target_column]), label=valid_data[target_column])

    params = {
        "objective" : "reg:pseudohubererror",
        "nthread": 4,
        "max_depth": max_depth, 
        "eval_metric": "mae",
        "learning_rate": 0.017,
        "subsample": 0.9,
        "colsample_bytree": 0.6, 
        "tree_method": "hist",

#        "" : 'hist'
    }
    model = xgb.train(params, train_matrix, num_boost_round=n_trees, evals=[(valid_matrix, "validation")], verbose_eval=100)
    val_loss = model.eval(valid_matrix)
    print(f"Validation loss: {val_loss}")
    return float(val_loss.split(":")[1]) #{'loss' :float(val_loss.split(":")[1])}



@workflow
def main_xgboost_workflow(parquet_path: str, num_samples: int = 10, max_parallel_trials: int = 3) -> dict:
    return hyperopt_with_xgboost_workflow(parquet_path=parquet_path, num_samples=num_samples, max_parallel_trials=max_parallel_trials)

@workflow
def practice_workflow(parquet_path: FlyteFile) -> bool:
    data = load_polars_data(input_file=parquet_path)
    print_df_stats(df=data)
    train_xgboost_polars(traintest=data, n_trees=1000, max_depth=12)
    return True
