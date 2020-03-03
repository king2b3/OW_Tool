import pandas as pd

data = pd.read_csv("InData.csv", dtype=object)

for row in data.iterrows():
    print(row)