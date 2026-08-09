import pandas as pd
from sklearn.model_selection import train_test_split
filtered_dg = pd.read_csv("/Users/aminkaabi/Downloads/filtered_dg.csv")
filtered_di = pd.read_csv("/Users/aminkaabi/Downloads/filtered_di.csv")

target_col = "SeriousDlqin2yrs"



def get_features(filtered_dg, target_col):
    y = filtered_dg['SeriousDlqin2yrs']
    X = filtered_dg.drop(columns=["SeriousDlqin2yrs"])
    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_preds_features(df):
    X_test_final = df.drop(columns=["SeriousDlqin2yrs"], errors="ignore").copy()
    return X_test_final