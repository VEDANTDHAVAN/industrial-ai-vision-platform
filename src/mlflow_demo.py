import mlflow

with mlflow.start_run():

    mlflow.log_param(
        "learning_rate",
        0.001
    )

    mlflow.log_param(
        "batch_size",
        16
    )

    mlflow.log_metric(
        "accuracy",
        99.55
    )

    mlflow.log_metric(
        "loss",
        0.0083
    )

print("Run logged.")