import pandas as pd
from src.train import train_model_b, train_model_a, predict_model_a, predict_model_b
from src.features import get_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

path = "filtered_dg.csv"
filtered_dg = pd.read_csv(path)


X, y = get_features(filtered_dg, "SeriousDlqin2yrs")

def evaluate_model_a(model_a, X_test, y_test):
    pred_a, proba_a = predict_model_a(model_a, X_test)

    print("AUC A:", roc_auc_score(y_test, proba_a))
    print(confusion_matrix(y_test, pred_a))
    print(classification_report(y_test, pred_a))


def evaluate_model_b(model_b, X_test, y_test):
    pred_b, proba_b = predict_model_b(model_b, X_test)

    print("AUC B:", roc_auc_score(y_test, proba_b))
    print(confusion_matrix(y_test, pred_b))
    print(classification_report(y_test, pred_b))






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model_a = train_model_a(X_train, y_train)
evaluate_model_a(model_a, X_test, y_test)
model_b = train_model_b(X_train, y_train)
evaluate_model_b(model_b, X_test, y_test)

proba_train = model_b.predict_proba(X_train)[:, 1]
proba_val = model_b.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, proba_train)
auc_val = roc_auc_score(y_train, proba_val)
print("verification overfitting:")
print("AUC train :", auc_train)
print("AUC validation :", auc_val)
print("Écart :", auc_train - auc_val)
