import sys
import os

import pandas as pd
import numpy as np

from logger import logging
from exception import CustomException
from utils import load_object, save_object
from build_feature import BuildFeature

from dataclasses import dataclass


@dataclass
class PredictPipelineConfig:
    modelPath: str = "model.pkl"


class PredictPipeline:
    def __init__(self):
        self.ingestion_config = PredictPipelineConfig()

    def Prediction(self, df):
        try:

            logging.info("Processing Data ........")
            dataset = BuildFeature().initiate_Feature_Building(df)

            logging.info("Predicting.........")
            model = load_object(file_path=self.ingestion_config.modelPath)

            preds = model.predict(dataset)

            df["prediction"] = preds

            logging.info("Result Generated")

            return preds

        except Exception as e:
            raise CustomException(e, sys)
