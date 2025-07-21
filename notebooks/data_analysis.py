import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/raw/uber.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:\n", df.info())
print("\nSummary Stats:\n", df.describe())
# Drop missing values:
df = df.dropna()
print("Nulls remaining:\n", df.isnull().sum())
#  Convert pickup time to datetime:
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
# Drop weird/negative fares or distances
df = df[df['fare_amount'] > 0]
#  cleaned file:
df.to_csv("data/cleaned/uber_cleaned.csv", index=False)
# EDA — Descriptive Stats + Visuals
print("Mean fare:", df['fare_amount'].mean())
print("Median fare:", df['fare_amount'].median())
print("Standard deviation:", df['fare_amount'].std())

# Histogram of fare amount:
sns.histplot(df['fare_amount'], bins=40, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare ($)")
plt.ylabel("Count")
plt.savefig("powerbi/screenshots/fare_distribution.png")
plt.clf()

# Outlier detection (IQR):

Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['fare_amount'] < Q1 - 1.5 * IQR) | (df['fare_amount'] > Q3 + 1.5 * IQR)]
print("Outliers:", outliers.shape[0])


df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['weekday'] = df['pickup_datetime'].dt.day_name()

# Peak time: 7-9 AM or 5-7 PM
df['peak_time'] = df['hour'].apply(lambda x: 'Peak' if (7 <= x <= 9) or (17 <= x <= 19) else 'Off-Peak')
# final enhanced dataset:
df.to_csv("data/enhanced/uber_enhanced.csv", index=False)
