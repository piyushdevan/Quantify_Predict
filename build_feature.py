import os
import sys


from logger import logging
from exception import CustomException
from utils import (
    LowPassFilter,
    PrincipalComponentAnalysis,
    NumericalAbstraction,
    FourierTransformation,
)
import pandas as pd
import numpy as np


from dataclasses import dataclass


@dataclass
class BuildFeaturesConfig:
    pass


class BuildFeature:
    def __init__(self):
        self.outlier_config = BuildFeaturesConfig()

    def initiate_Feature_Building(self, df):
        try:
            LowPass = LowPassFilter()
            PCA = PrincipalComponentAnalysis()
            NumAbs = NumericalAbstraction()
            FreqAbs = FourierTransformation()

            logging.info("start feature building")

            # df.set_index("epoch (ms)", inplace=True)
            predictor_columns = list(df.columns[0:6])

            for col in predictor_columns:
                df[col] = df[col].interpolate()

            fs = 1000 / 200
            fc = 1.2

            # logging.info("Applying Lowpass")
            for col in predictor_columns:
                df[col] = LowPass.low_pass_filter(df[col], fs, fc, order=5)

            # logging.info("Applying pca")

            pc_values = PCA.determine_pc_explained_variance(df, predictor_columns)
            df = PCA.apply_pca(df, predictor_columns, 3)

            # logging.info("Adding rms acc and rms gyr")

            acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
            gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
            df["acc_r"] = np.sqrt(acc_r)
            df["gyr_r"] = np.sqrt(gyr_r)

            # logging.info("Adding temporal abstraction")
            NumAbs = NumericalAbstraction()
            predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

            ws = int(1000 / 200)

            for col in predictor_columns:
                df = NumAbs.abstract_numerical(df, [col], ws, "mean")
                df = NumAbs.abstract_numerical(df, [col], ws, "std")

            cols = df.filter(like="_temp_", axis=1)
            for col in cols:
                df[col].fillna(df[col].mean(), inplace=True)

            # logging.info("Adding Fourier_transform")
            df = df.reset_index()
            FreqAbs = FourierTransformation()

            fs = int(1000 / 200)
            ws = int(2800 / 200)

            df = FreqAbs.abstract_frequency(df, predictor_columns, ws, fs)

            cols = df.filter(like="_freq_", axis=1)
            for col in cols:
                df[col].fillna(df[col].mean(), inplace=True)

            df = df.set_index("epoch (ms)")
            logging.info("Feature Building Done")

            return df

        except Exception as e:
            raise CustomException(e, sys)
