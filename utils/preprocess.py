"""
preprocess.py

Preprocessing utilities for REAL Bengaluru housing dataset
augmented with infrastructure and location features.

This file ensures CONSISTENT preprocessing
between training and future inference.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --------------------------------------------------
# NUMERIC COLUMNS USED FOR CLUSTERING
# --------------------------------------------------
NUM_COLS = [
    "price_inr",
    "airport_distance_km",
    "railway_station_distance_km",
    "nearest_hospital_distance_km",
    "hospitals_count",
    "schools_count",
]

# --------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess_features(df: pd.DataFrame):
    """
    Preprocess housing data for clustering:
    - select numeric features
    - log-transform price (dominant signal)
    - median imputation
    - return processed dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Raw housing dataframe

    Returns
    -------
    X : pd.DataFrame
        Preprocessed numeric feature matrix
    """

    X = df[NUM_COLS].copy()

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    X = X.fillna(X.median())

    # -----------------------------
    # LOG TRANSFORM PRICE
    # (CRITICAL FOR CLUSTER QUALITY)
    # -----------------------------
    X["price_inr"] = np.log1p(X["price_inr"])

    return X


# --------------------------------------------------
# LOCATION ENCODING (OPTIONAL HELPER)
# --------------------------------------------------
def encode_location(df: pd.DataFrame, column: str = "location"):
    """
    Encode location column numerically.

    Returns:
    - encoded array
    - fitted LabelEncoder
    """
    le = LabelEncoder()
    encoded = le.fit_transform(df[column].astype(str))
    return encoded, le


# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------
def scale_features(X: pd.DataFrame):
    """
    Scale features using StandardScaler.

    Returns:
    - scaled numpy array
    - fitted scaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler









# """
# preprocess.py

# This file defines preprocessing utilities and feature configuration
# used consistently across training and prediction.

# Even though this file does not train a model directly,
# it plays a CRITICAL role in ensuring consistency.
# """

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # -----------------------------
# # Numeric columns used for clustering
# # -----------------------------
# NUM_COLS = [
#     "total_units",
#     "median_price",
#     "median_rent",
#     "schools_count",
#     "nearest_school_distance_km",
#     "school_score",
#     "hospitals_count",
#     "nearest_hospital_distance_km",
#     "hospital_score",
#     "population_density",
# ]

# def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Apply preprocessing steps to raw dataframe:
#     - select numeric columns
#     - handle missing values
#     - apply log transformations to skewed features

#     Parameters:
#     df (pd.DataFrame): Raw housing dataframe

#     Returns:
#     pd.DataFrame: Preprocessed dataframe
#     """

#     X = df[NUM_COLS].copy()

#     # Handle missing values using median imputation
#     X = X.fillna(X.median())

#     # Reduce skewness
#     X["median_price"] = np.log1p(X["median_price"])
#     X["median_rent"] = np.log1p(X["median_rent"])
#     X["population_density"] = np.log1p(X["population_density"])

#     return X


# def scale_features(X: pd.DataFrame):
#     """
#     Scale features using StandardScaler.

#     Returns:
#     - scaled array
#     - fitted scaler object
#     """
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     return X_scaled, scaler
