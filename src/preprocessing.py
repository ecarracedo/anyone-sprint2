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

    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.

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
    #   - In order to prevent ovrfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    # Encode categorical features
    working_train_df, working_test_df, working_val_df = encode_dataset(working_train_df,
                                                                       working_test_df,
                                                                       working_val_df)
    # Simple Imputer
    working_train_df, working_test_df, working_val_df = simpleimputer_dataset(working_train_df,
                                                                       working_test_df,
                                                                       working_val_df)

    # MinMax Scaler
    working_train_df, working_test_df, working_val_df = mixmax_dataset(working_train_df,
                                                                       working_test_df,
                                                                       working_val_df)

    # Dataframe to Numpy
    working_train_ndarray = working_train_df.to_numpy()
    working_test_ndarray = working_test_df.to_numpy()
    working_val_ndarray = working_val_df.to_numpy()

    return working_train_ndarray, working_val_ndarray, working_test_ndarray


def encode_dataset(df, df_test, df_val):

    ############################ OrdinalEncoder ###########################

    # Identificar columnas categóricas con valores binarios o hasta 2 categorías
    cat_cols = df.select_dtypes(include='object').columns

    # Filtrar las columnas categóricas que tienen 2 categorías y aplicar OrdinalEncoder
    binary_cat_cols = [col for col in cat_cols if df[col].nunique() == 2]

    # Aplicar OrdinalEncoder a las columnas binarias
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    df[binary_cat_cols] = encoder.fit_transform(df[binary_cat_cols])

    ########################### OneHotEncoder ##############################

    # Identificar columnas categóricas con más de 2 categorías
    multi_cat_cols = [col for col in cat_cols if df[col].nunique() > 2]

    # Aplicar OneHotEncoder
    # Usamos sparse_output=False para obtener un DataFrame, no una matriz dispersa
    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
    onehot_encoded = onehot_encoder.fit_transform(df[multi_cat_cols])

    ########################### Unir Encoders ##############################

    # Crear DataFrame con los nuevos nombres de columnas
    onehot_feature_names = onehot_encoder.get_feature_names_out(multi_cat_cols)
    df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_feature_names, index=df.index)

    # Eliminar las columnas originales y añadir las nuevas codificadas
    df = df.drop(columns=multi_cat_cols)
    df = pd.concat([df, df_onehot], axis=1)

    # Aplicar los encoders ya entrenados
    df_val[binary_cat_cols] = encoder.transform(df_val[binary_cat_cols])
    df_test[binary_cat_cols] = encoder.transform(df_test[binary_cat_cols])
    onehot_encoded_val = onehot_encoder.transform(df_val[multi_cat_cols])
    onehot_encoded_test = onehot_encoder.transform(df_test[multi_cat_cols])

    df_onehot_val = pd.DataFrame(onehot_encoded_val, columns=onehot_feature_names, index=df_val.index)
    df_onehot_test = pd.DataFrame(onehot_encoded_test, columns=onehot_feature_names, index=df_test.index)
    
    df_val = df_val.drop(columns=multi_cat_cols)
    df_val = pd.concat([df_val, df_onehot_val], axis=1)

    df_test = df_test.drop(columns=multi_cat_cols)
    df_test = pd.concat([df_test, df_onehot_test], axis=1)

    return df, df_test, df_val

def simpleimputer_dataset(df, df_test, df_val):
    # Create an instance of SimpleImputer with median strategy
    imputer = SimpleImputer(strategy='median')
    
    # Fit the imputer on the training data
    imputer.fit(df)

    # Transform the training, validation, and test datasets
    imputed_train = pd.DataFrame(imputer.transform(df), columns=df.columns)
    imputed_test = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)
    imputed_val = pd.DataFrame(imputer.transform(df_val), columns=df_val.columns)
    
    return imputed_train, imputed_test, imputed_val

def mixmax_dataset(df, df_test, df_val):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data
    scaler.fit(df)

    # Transform the training, validation, and test datasets
    scaled_train = pd.DataFrame(scaler.transform(df), columns=df.columns)
    scaled_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    scaled_val = pd.DataFrame(scaler.transform(df_val), columns=df_val.columns)

    return scaled_train, scaled_test, scaled_val