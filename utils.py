import os
import sys
import scipy
import math
import scipy.special
from scipy.signal import butter, lfilter, filtfilt
import scipy.stats as stats


import numpy as np
import pandas as pd

# import dill
import pickle

import copy

from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from exception import CustomException


#####################################################################
###                    GENERAL UTILITY FUNCTION                   ###
#####################################################################


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(f"training mode {model}")
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5, scoring="accuracy")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)

            test_model_score = accuracy_score(y_test, y_test_pred)
            print(test_model_score, train_model_score)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


#####################################################################
###             FEATURE ENGINEERING FUNCTION AND CLASSES          ###
#####################################################################


def mark_outliers_chauvenet(dataset, col, C=2):
    try:
        dataset = dataset.copy()
        # Compute the mean and standard deviation.
        mean = dataset[col].mean()
        std = dataset[col].std()
        N = len(dataset.index)
        criterion = 1.0 / (C * N)

        # Consider the deviation for the data points.
        deviation = abs(dataset[col] - mean) / std

        # Express the upper and lower bounds.
        low = -deviation / math.sqrt(C)
        high = deviation / math.sqrt(C)
        prob = []
        mask = []

        # Pass all rows in the dataset.
        for i in range(0, len(dataset.index)):
            # Determine the probability of observing the point
            prob.append(
                1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
            )
            # And mark as an outlier when the probability is below our criterion.
            mask.append(prob[i] < criterion)
        dataset[col + "_outlier"] = mask
        return dataset

    except Exception as e:
        raise CustomException(e, sys)


class LowPassFilter:
    def low_pass_filter(
        self,
        data_col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_col = filtfilt(b, a, data_col)
        else:
            data_col = lfilter(b, a, data_col)
        return data_col


class PrincipalComponentAnalysis:
    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    # Perform the PCA on the selected columns and return the explained variance.
    def determine_pc_explained_variance(self, data_table, cols):
        # Normalize the data first.
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        # And return the explained variances.
        return self.pca.explained_variance_ratio_

    def apply_pca(self, data_table, cols, number_comp):
        dt_norm = self.normalize_dataset(data_table, cols)

        # perform the PCA.
        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        # Transform our old values.
        new_values = self.pca.transform(dt_norm[cols])

        # And add the new ones:
        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table


class NumericalAbstraction:
    def aggregate_value(self, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    def abstract_numerical(self, data_table, cols, window_size, aggregation_function):
        # Create new columns for the temporal data, pass over the dataset and compute values
        for col in cols:
            data_table[
                col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            ] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        return data_table


class FourierTransformation:
    def find_fft_transformation(self, data, sampling_rate):
        transformation = np.fft.rfft(data, len(data))
        return transformation.real, transformation.imag

    def abstract_frequency(self, data_table, cols, window_size, sampling_rate):
        freqs = np.round((np.fft.rfftfreq(int(window_size)) * sampling_rate), 3)

        for col in cols:
            data_table[col + "_freq_max"] = np.nan
            data_table[col + "_freq_weighted"] = np.nan
            data_table[col + "_freq_pse"] = np.nan
            for freq in freqs:
                data_table[
                    col + "_freq_" + str(freq) + "_Hz_ws_" + str(window_size)
                ] = np.nan

        for i in range(window_size, len(data_table.index)):
            for col in cols:
                real_ampl, imag_ampl = self.find_fft_transformation(
                    data_table[col].iloc[
                        i - window_size : min(i + 1, len(data_table.index))
                    ],
                    sampling_rate,
                )
                # We only look at the real part in this implementation.
                for j in range(0, len(freqs)):
                    data_table.loc[
                        i, col + "_freq_" + str(freqs[j]) + "_Hz_ws_" + str(window_size)
                    ] = real_ampl[j]

                data_table.loc[i, col + "_freq_max"] = freqs[
                    np.argmax(real_ampl[0 : len(real_ampl)])
                ]
                data_table.loc[i, col + "_freq_weighted"] = float(
                    np.sum(freqs * real_ampl)
                ) / np.sum(real_ampl)
                PSD = np.divide(np.square(real_ampl), float(len(real_ampl)))
                PSD_pdf = np.divide(PSD, np.sum(PSD))
                data_table.loc[i, col + "_freq_pse"] = -np.sum(
                    np.log(PSD_pdf) * PSD_pdf
                )

        return data_table
