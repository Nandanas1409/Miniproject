import numpy as np
import pandas as pd

def preprocess_depression_data(raw_df, scaler):
    """
    Cleans and scales depression EEG feature data.
    Returns scaled feature array.
    """

    # Drop non-numeric columns
    non_numeric = raw_df.select_dtypes(include=["object"]).columns
    df = raw_df.drop(columns=non_numeric)

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop completely empty columns
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaN
    df = df.fillna(df.median())

    # Scale
    X_scaled = scaler.transform(df)

    return X_scaled
