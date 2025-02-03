from typing import List

import numpy as np
import pandas as pd
from pyspark.sql.functions import col, when

from db import spark


def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess the PBC survival dataset using Apache Spark.

    This function reads a CSV file into a Spark DataFrame, performs necessary preprocessing steps,
    and converts it to a Pandas DataFrame for further analysis.

    Processing Steps:
    - Removes the 'id' column (irrelevant for modeling)
    - Filters out rows where `status == 1`
    - Converts `status` to binary (status / 2.0)
    - Normalizes `time` from days to years (time / 365.0)
    - Adjusts `trt` column to zero-indexed (trt - 1)
    - Encodes `sex` column ('f' → 0.0, 'm' → 1.0)
    - Drops rows with missing values

    Args:
        path (str): File path to the CSV dataset.

    Returns:
        pd.DataFrame: Preprocessed Pandas DataFrame.
    """
    # Load CSV into Spark DataFrame
    df = spark.read.csv(path, header=True, inferSchema=True)

    # Data preprocessing
    df = (
        df.drop("id")
        .filter(col("status") != 1)
        .withColumn("status", col("status") / 2.0)
        .withColumn("time", col("time") / 365.0)
        .withColumn("trt", col("trt") - 1)
        .withColumn("sex", when(col("sex") == "f", 0.0).when(col("sex") == "m", 1.0))
        .dropna()
    )

    return df.toPandas()


def to_one_hot(dataframe: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert columns in dataframe to one-hot encoding.
    Args:
        dataframe (dataframe): pandas dataframe containing covariates
        columns (list of strings): list categorical column names to one hot encode
    Returns:
        one_hot_df (dataframe): dataframe with categorical columns encoded
                            as binary variables
    """

    one_hot_df = pd.get_dummies(
        dataframe, columns=columns, drop_first=True, dtype=np.float64
    )

    return one_hot_df


def hazard_ratio(
        case_1: np.ndarray, case_2: np.ndarray, cox_params: np.ndarray
) -> float:
    """
    Return the hazard ratio of case_1 : case_2 using
    the coefficients of the cox model.

    Args:
        case_1 (np.array): (1 x d) array of covariates
        case_2 (np.array): (1 x d) array of covariates
        model (np.array): (1 x d) array of cox model coefficients
    Returns:
        hazard_ratio (float): hazard ratio of case_1 : case_2
    """

    hr = np.exp(cox_params.dot((case_1 - case_2).T))

    return hr


def harrell_c(y_true: np.ndarray, scores: np.ndarray, event: np.ndarray) -> float:
    """
    Compute Harrel C-index given true event/censoring times,
    model output, and event indicators.

    Args:
        y_true (array): array of true event times
        scores (array): model risk scores
        event (array): indicator, 1 if event occurred at that index, 0 for censorship
    Returns:
        result (float): C-index metric
    """

    n = len(y_true)
    assert len(scores) == n and len(event) == n

    concordant = 0.0
    permissible = 0.0
    ties = 0.0

    result = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 1 or event[j] == 1:
                if event[i] == 1 and event[j] == 1:
                    permissible += 1.0
                    if scores[i] == scores[j]:
                        ties += 1.0
                    elif y_true[i] < y_true[j] and scores[i] > scores[j]:
                        concordant += 1.0
                    elif y_true[i] > y_true[j] and scores[i] < scores[j]:
                        concordant += 1.0
                elif event[i] != event[j]:
                    censored = j
                    uncensored = i
                    if event[i] == 0:
                        censored = i
                        uncensored = j
                    if y_true[uncensored] <= y_true[censored]:
                        permissible += 1.0
                        if scores[uncensored] == scores[censored]:
                            ties += 1.0
                        if scores[uncensored] > scores[censored]:
                            concordant += 1.0

    result = (concordant + 0.5 * ties) / permissible

    return result
