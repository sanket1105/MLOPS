## just we are trying to do the MLFLow on the AWS - EC2 and not deploy everything there

import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

os.environ["MLFLOW_TRACKING_URI"] = (
    "http://ec2-54-174-51-81.compute-1.amazonaws.com:5000"
)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))

    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return rmse, mae, r2


if __name__ == "__main__":

    data = pd.read_csv("winequality-red.csv")

    train, test = train_test_split(data)

    train_X = train.drop(["quality"], axis=1)
    test_X = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    ## elastic net: lets run this
    # if argument given then take that value else by default 0.5

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_X, train_y)

        y_pred = lr.predict(test_X)
        (rmse, mae, r2) = eval_metrics(test_y, y_pred)

        print(f"Elasticnet model with alpha:{alpha} and l1_ratio: {l1_ratio}")
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        ## AWS remote server setup

        remote_server_uri = (
            "http://ec2-54-174-51-81.compute-1.amazonaws.com:5000"  # AWS server
        )

        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
