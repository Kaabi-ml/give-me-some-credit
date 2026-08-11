import pandas as pd
import joblib
from src.features import get_features, get_preds_features

path = "filtered_di.csv"
output_path = "submission.csv"

df = pd.read_csv(path)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

X, y = get_features(df, "SeriousDlqin2yrs")
X_test_final = get_preds_features(df)

# chargement du modèle déjà entraîné
model_b = joblib.load("model_b.joblib")

# prédiction des probabilités
probs = model_b.predict_proba(X_test_final)[:, 1]

# création du fichier de soumission


submission = pd.DataFrame({
    "Id": range(1, len(probs) + 1),
    "Probability": probs
})

submission.to_csv(output_path, index=False)


print("submission.csv créé avec succès")

sample = pd.read_csv("GiveMeSomeCredit/sampleEntry.csv")
submission = pd.read_csv(output_path)




