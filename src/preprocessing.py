from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_train_df["EMERGENCYSTATE_MODE"].fillna("No", inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.

    #Identificar columnas categóricas con valores binarios o hasta 2 categorías
    cat_cols = working_train_df.select_dtypes(include='object').columns

    #Filtrar las columnas categóricas que tienen 2 categorías y aplicar OrdinalEncoder
    binary_cat_cols = [col for col in cat_cols if working_train_df[col].nunique() == 2]
    encoder = OrdinalEncoder()
    working_train_df[binary_cat_cols] = encoder.fit_transform(working_train_df[binary_cat_cols])

    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.

    #Identificar columnas categóricas que NO son binarias o mayor a 2 categorías
    multi_cat_cols = [col for col in cat_cols if working_train_df[col].nunique() > 2]

    # Aplicar OneHotEncoder
    # Usamos sparse_output=False para obtener un DataFrame, no una matriz dispersa
    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
    onehot_encoded = onehot_encoder.fit_transform(working_train_df[multi_cat_cols])

    #Crear DataFrame con los nuevos nombres de columnas
    onehot_feature_names = onehot_encoder.get_feature_names_out(multi_cat_cols)
    df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=working_train_df.index)

    # Eliminar las columnas originales y añadir las nuevas codificadas
    working_train_df = working_train_df.drop(columns=multi_cat_cols)
    working_train_df = pd.concat([working_train_df, df_onehot], axis=1)


    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    
    


    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.


    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.


    return working_train_df
