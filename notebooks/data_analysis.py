import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../data/raw/uber.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:\n", df.info())
print("\nSummary Stats:\n", df.describe())

df = df.dropna()
print("Nulls remaining:\n", df.isnull().sum())
