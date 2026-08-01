import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame, scaler, model_columns) -> "np.ndarray":
    df = df.copy()

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    binary_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
    df['Gender'] = df['Gender'].map(binary_map)
    df['Married'] = df['Married'].map(binary_map)
    df['Self_Employed'] = df['Self_Employed'].map(binary_map)

    df = pd.get_dummies(df, columns=['Education', 'Property_Area'], drop_first=True)

    missing_features = [f for f in model_columns if f not in df.columns]

    df_aligned = df.reindex(columns=model_columns, fill_value=0)
    scaled = scaler.transform(df_aligned)

    return scaled, missing_features