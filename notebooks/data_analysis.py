import pandas as pd

# Load the raw Excel file
df = pd.read_excel("data/raw/Plant_DTS.xls")

# Preview the data
print(df.head())
print(df.info())
