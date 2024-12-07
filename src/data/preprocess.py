import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])
    return data
