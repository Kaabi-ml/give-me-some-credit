import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.features import get_features

filtered_dg = pd.read_csv("/Users/aminkaabi/Downloads/filtered_dg.csv")

X, y = get_features(filtered_dg, "SeriousDlqin2yrs")
# model without balanced
def build_model_a():
    model_a = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    return model_a

def train_model_a(X_train, y_train):
    model_a = build_model_a()
    model_a.fit(X_train, y_train)
    return model_a

def predict_model_a(model_a, X):
    preds_a = model_a.predict(X)
    probs_a = model_a.predict_proba(X)[:, 1]
    return preds_a, probs_a




# modèle avec balanced
def build_model_b():
    model_b = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )
    return model_b

def train_model_b(X_train, y_train):
    model_b = build_model_b()
    model_b.fit(X_train, y_train)
    return model_b

def predict_model_b(model_b, X):
    preds_b = model_b.predict(X)
    probs_b = model_b.predict_proba(X)[:, 1]
    return preds_b, probs_b






# compare and evaluate models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_b = train_model_b(X_train, y_train)

print("Model B trained successfully")
print("Intercept:", model_b.named_steps["logisticregression"].intercept_)
print("Coef shape:", model_b.named_steps["logisticregression"].coef_.shape)


import joblib

model_b = train_model_b(X_train, y_train)
joblib.dump(model_b, "/Users/aminkaabi/Downloads/model_b.joblib")
print("Model saved to Downloads")