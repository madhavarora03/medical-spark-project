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
