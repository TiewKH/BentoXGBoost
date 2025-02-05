import os

import numpy as np
import xgboost as xgb

import bentoml
import os
from bentoml.models import BentoModel


model_name = os.environ["MODEL"]

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class CancerTiewKHClassifier:
    # Retrieve the latest version of the model from the BentoML model store

    # This fails
    bento_model = [BentoModel(f"{model_name}:latest")]

    # This works
    # bento_model = BentoModel(f"{model_name}:latest")

    def __init__(self):
        self.model = xgb.Booster({'nthread': 4})  # init model
        # This fails
        self.model.load_model(f"{self.bento_model[0].stored.path}/cancer-tiewkh-test.model")

        # This works
        # self.model.load_model(f"{self.bento_model.stored.path}/cancer-tiewkh-test.model")

        # Check resource availability
        if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
            self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
        else:
            nthreads = os.getenv("OMP_NUM_THREADS")
            if nthreads:
                nthreads = max(int(nthreads), 1)
            else:
                nthreads = 1
            self.model.set_param({"predictor": "cpu_predictor", "nthread": nthreads})

    @bentoml.api
    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(xgb.DMatrix(data))
