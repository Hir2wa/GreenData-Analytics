import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats

# Load data from raw folder
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