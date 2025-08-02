import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
# # Load data from raw folder
plant_fact_path = r'C:\Users\Aime\Desktop\BigData - Exam\data\raw\Plant_DTS.xls'
accounts_path = r'C:\Users\Aime\Desktop\BigData - Exam\data\raw\AccountAnalysed.xlsx'
plant_hierarchy_path = r'C:\Users\Aime\Desktop\BigData - Exam\data\raw\Plant_Hearchy.xlsx'

df_fact = pd.read_excel(plant_fact_path, sheet_name='Plant_FACT')
df_accounts = pd.read_excel(accounts_path)
df_hierarchy = pd.read_excel(plant_hierarchy_path)

# Merge datasets on common keys
df = df_fact.merge(df_accounts, on='Account_id', how='left')  # Merge with Accounts
df = df.merge(df_hierarchy, left_on='Product_id', right_on='Product_Name_id', how='left')

# Function to clean data
def clean_data(df):
    df = df.dropna()  # Handle missing values
    df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')  # Standardize date format
    # Remove outliers using IQR for quantity
    Q1 = df['quantity'].quantile(0.25)
    Q3 = df['quantity'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['quantity'] < (Q1 - 1.5 * IQR)) | (df['quantity'] > (Q3 + 1.5 * IQR)))]
    return df

# Function to enhance data
def enhance_data(df):
    df['Sales_Per_Unit'] = df['Sales_USD'] / df['quantity']  # New feature
    df = pd.get_dummies(df, columns=['country_code'], prefix='country_code')  # Encode country with prefix
    scaler = StandardScaler()
    df[['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD']] = scaler.fit_transform(df[['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD']])
    return df

# Function for EDA
def run_eda(df):
    # Detailed descriptive statistics
    print("Descriptive Statistics:")
    stats = df[['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD', 'Sales_Per_Unit']].describe()
    print(stats)
    print("\nAdditional Statistics:")
    print(f"Mean of Quantity: {df['quantity'].mean():.2f}")
    print(f"Median of Quantity: {df['quantity'].median():.2f}")
    print(f"Mode of Quantity: {stats.mode()['quantity'][0] if not df['quantity'].mode().empty else 'N/A'}")
    print(f"Skewness of Quantity: {df['quantity'].skew():.2f}")

    # Outlier detection and visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['quantity'])
    plt.title('Box Plot of Quantity (Outliers)')
    plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\quantity_boxplot.png')
    plt.close()

    # Distribution visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(df['quantity'], kde=True)
    plt.title('Quantity Distribution with KDE')
    plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\quantity_dist_kde.png')
    plt.close()

    # Relationships visualization
    country_cols = [col for col in df.columns if col.startswith('country_code_')]
    hue_col = country_cols[0] if country_cols else None
    if hue_col:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x='Sales_USD', y='quantity', hue=hue_col, data=df)
        plt.title(f'Sales USD vs Quantity by {hue_col}')
        plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\sales_vs_qty_country.png')
        plt.close()
    else:
        print("No country code columns found for hue.")

    # Correlation heatmap
    plt.figure(figsize=(10, 5))
    numeric_df = df[['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD', 'Sales_Per_Unit']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numeric Variables')
    plt.savefig(r'C:\Users\Aime\Desktop\BigData - Exam\data\cleaned\correlation_heatmap.png')
    plt.close()

# Function for clustering
def train_model(df):
    X = df[['quantity', 'Sales_USD']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f'\nSilhouette Score: {score:.4f}')
    df['Cluster'] = labels
    return df

# Main execution for processing
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

# Separate section for Power BI screenshots
output_folder = r"C:\Users\Aime\Desktop\BigData - Exam\powerbi\screenshots\\"

# Load enhanced dataset for screenshots
df_enhanced = pd.read_excel(r'C:\Users\Aime\Desktop\BigData - Exam\data\enhanced\enhanced_plant_fact.xlsx')

# 1. Quantity Box Plot
plt.figure(figsize=(8, 5))
plt.boxplot(df_enhanced['quantity'].dropna())
plt.title("Box Plot of Quantity")
plt.ylabel("Quantity")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{output_folder}quantity_boxplot.png", bbox_inches='tight')
plt.close()

# 2. Quantity Histogram + KDE
quantity = df_enhanced['quantity'].dropna()
plt.figure(figsize=(8, 5))
plt.hist(quantity, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
from scipy.stats import gaussian_kde
kde = gaussian_kde(quantity)
x_vals = np.linspace(quantity.min(), quantity.max(), 200)
plt.plot(x_vals, kde(x_vals), color='red', linewidth=2, label="KDE")
plt.title("Histogram + KDE of Quantity")
plt.xlabel("Quantity")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(f"{output_folder}quantity_dist_kde.png", bbox_inches='tight')
plt.close()

# 3. Correlation Heatmap
corr_cols = ['quantity', 'Sales_USD', 'Price_USD', 'COGS_USD', 'Sales_Per_Unit']
corr = df_enhanced[corr_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Heatmap")
plt.colorbar(label='Correlation Coefficient')

tick_marks = np.arange(len(corr.columns))
plt.xticks(tick_marks, corr.columns, rotation=45)
plt.yticks(tick_marks, corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        value = round(corr.iloc[i, j], 2)
        plt.text(j, i, str(value), ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig(f"{output_folder}correlation_heatmap.png", bbox_inches='tight')
plt.close()




# Load your Excel file
input_file = "data/enhanced/enhanced_plant_fact.xlsx"
output_dir = "powerbi/screenshots"
os.makedirs(output_dir, exist_ok=True)

# Load DataFrame
df = pd.read_excel(input_file)

# Quantity Distribution Plot
plt.figure(figsize=(8, 6))
sns.histplot(df['quantity'], bins=30, kde=False)
plt.title('Quantity Distribution')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.savefig(f"{output_dir}/quantity_dist.png")
plt.close()

# Quantity Distribution with KDE
plt.figure(figsize=(8, 6))
sns.histplot(df['quantity'], bins=30, kde=True)
plt.title('Quantity Distribution with KDE')
plt.xlabel('Quantity')
plt.ylabel('Density')
plt.savefig(f"{output_dir}/quantity_dist_kde.png")
plt.close()

# Quantity Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['quantity'])
plt.title('Boxplot of Quantity')
plt.savefig(f"{output_dir}/quantity_boxplot.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation = df.select_dtypes(include='number').corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

# Sales vs Quantity Scatter Plot
if 'Sales_USD' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='quantity', y='Sales_USD', data=df)
    plt.title('Sales vs Quantity')
    plt.xlabel('Quantity')
    plt.ylabel('Sales (USD)')
    plt.savefig(f"{output_dir}/sales_vs_qty.png")
    plt.close()
    


from sklearn.cluster import KMeans

# Load your enhanced dataset
df = pd.read_excel('data/enhanced/enhanced_plant_fact.xlsx')

# Strip any spaces from column names just in case
df.columns = df.columns.str.strip()

# === 1. 📊 Price per Unit vs Quantity Plot ===
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Sales_Per_Unit',    # Use the exact column name from your data
    y='quantity',          # lowercase 'quantity'
    hue='country2',        # your country column
    palette='tab10',
    alpha=0.7
)
plt.title('Price per Unit vs Quantity')
plt.xlabel('Price per Unit (USD)')
plt.ylabel('Quantity Sold')
plt.grid(True)
plt.tight_layout()
plt.savefig('powerbi/screenshots/price_vs_quantity.png')
plt.close()

# === 2. 🌍 Total Quantity Sold per Country ===
plt.figure(figsize=(10, 6))
country_summary = df.groupby('country2')['quantity'].sum().sort_values(ascending=False)
sns.barplot(
    x=country_summary.values,
    y=country_summary.index,
    palette='viridis'
)
plt.title('Total Quantity Sold per Country')
plt.xlabel('Quantity')
plt.ylabel('Country')
plt.tight_layout()
plt.savefig('powerbi/screenshots/quantity_per_country.png')
plt.close()

# === 3. 🤖 K-Means Clustering (Sales USD vs Quantity) ===
# Select relevant columns and drop missing values
cluster_df = df[['Sales_USD', 'quantity']].dropna()

# Fit KMeans clustering model
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_df['Cluster'] = kmeans.fit_predict(cluster_df)

# Plot clusters with centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=cluster_df,
    x='Sales_USD',
    y='quantity',
    hue='Cluster',
    palette='Set2'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c='red',
    label='Centroids',
    marker='X'
)
plt.title('K-Means Clustering (Sales USD vs Quantity)')
plt.xlabel('Sales (USD)')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('powerbi/screenshots/kmeans_sales_quantity.png')
plt.close()
print("✅ All plots saved successfully in 'powerbi/screenshots/' 🎉")






print("these is the start data visualization ")

# We'll cluster using Sales_USD and quantity
cluster_df = df[['Sales_USD', 'quantity']].dropna()

# Optional: Normalize data to improve clustering performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)
from sklearn.cluster import KMeans

# Initialize and train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the dataframe
cluster_df['Cluster'] = clusters

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=scaled_data[:, 0],  # Scaled Sales_USD
    y=scaled_data[:, 1],  # Scaled quantity
    hue=cluster_df['Cluster'],
    palette='Set2'
)

# Plot centroids (scaled back)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=200, c='red', marker='X', label='Centroids'
)

plt.title('KMeans Clustering (Sales vs Quantity)')
plt.xlabel('Scaled Sales USD')
plt.ylabel('Scaled Quantity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('powerbi/screenshots/kmeans_scaled.png')
plt.close()

score = silhouette_score(cluster_df[['Sales_USD', 'quantity']], clusters)
print(f"🧪 Silhouette Score: {score:.3f}")




# Load data (replace this with your own path)
df = pd.read_excel("data/enhanced/enhanced_plant_fact.xlsx")



# Select features for clustering
X = df[['Sales_USD', 'quantity']]

# Normalize the features (super important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try k from 2 to 10
k_values = list(range(2, 11))
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"k={k} ➤ Silhouette Score = {score:.3f}")

# Plot Elbow Method
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Distortion)')
plt.title('🔺 Elbow Method')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('📈 Silhouette Score for k')

plt.tight_layout()
plt.show()
