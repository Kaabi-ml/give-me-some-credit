import pandas as pd

path1 = "GiveMeSomeCredit/sampleEntry.csv"
df = pd.read_csv(path1)

path2 = "GiveMeSomeCredit/cs-training.csv"
dg = pd.read_csv(path2)
filtered_dg = dg.dropna(axis=0)

path3 = "GiveMeSomeCredit/cs-test.csv"
di = pd.read_csv(path3)
cols = [c for c in di.columns if c != "SeriousDlqin2yrs"]
filtered_di = di.fillna(di.median(numeric_only=True))
filtered_dg = filtered_dg.drop(columns=["Unnamed: 0"])




filtered_dg.to_csv("filtered_dg.csv", index=False)
filtered_di.to_csv("filtered_di.csv", index=False)

print(filtered_dg.shape)
print(filtered_di.shape)