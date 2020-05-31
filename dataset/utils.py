import numpy as np
import pandas as pd


def get_adult_income_raw_dataset(path):
    label_col = "income"
    df_raw_cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", label_col]
    df_raw = pd.read_csv(path, header=None, names=df_raw_cols, na_values=['?', ' ?'], skipinitialspace=True)
    if pd.isnull(df_raw.at[0, label_col]):  # test broken format
        df_raw = df_raw.drop(0)  # remove first row
        df_raw[label_col] = df_raw[label_col].str.replace(".", "")
    return df_raw
