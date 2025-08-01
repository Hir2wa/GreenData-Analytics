import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data from raw folder
file_path = r'C:\Users\Aime\Desktop\BigData - Exam\data\raw\Plant_DTS.xls'
df = pd.read_excel(file_path, sheet_name='Plant_FACT')  # Adjust sheet name as needed

# Function to clean data
def clean_data(df):
    df = df.dropna()  # Handle missing values
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Standardize date format
    # Remove outliers using IQR
    Q1 = df['Quantity'].quantile(0.25)
    Q3 = df['Quantity'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Quantity'] < (Q1 - 1.5 * IQR)) | (df['Quantity'] > (Q3 + 1.5 * IQR)))]
    return df

# Function to enhance data
def enhance_data(df):
    df['Qty_Change_Pct'] = df['Quantity'].pct_change() * 100  # Percentage change
    df = pd.get_dummies(df, columns=['Country'], drop_first=True)  # Encode countries
    scaler = StandardScaler()
    df[['Quantity', 'Value YTD']] = scaler.fit_transform(df[['Quantity', 'Value YTD']])
    return df

# Function for EDA
def run_eda(df):
    print(df.describe())
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Quantity'])
    plt.title('Quantity Distribution')
    plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\quantity_dist.png')
    plt.close()
    sns.scatterplot(x='Value YTD', y='Quantity', data=df)
    plt.title('Value YTD vs Quantity')
    plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\value_vs_qty.png')
    plt.close()

# Function for clustering
def train_model(df):
    X = df[['Quantity', 'Value YTD']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'Silhouette Score: {score}')
    df['Cluster'] = labels
    return df

# Main execution
if __name__ == "__main__":
    # Clean data
    cleaned_df = clean_data(df)
    cleaned_df.to_excel(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\cleaned_plant_fact.xlsx', index=False)

    # Enhance data
    enhanced_df = enhance_data(cleaned_df)
    enhanced_df.to_excel(r'C:\Users\Aime\Desktop\BigData - Exam\data\enhanced\enhanced_plant_fact.xlsx', index=False)

    # Run EDA
    run_eda(cleaned_df)

    # Train and evaluate model
    final_df = train_model(enhanced_df)
    final_df.to_excel(r'C:\Users\Aime\Desktop\BigData - Exam\data\enhanced\clustered_plant_fact.xlsx', index=False)