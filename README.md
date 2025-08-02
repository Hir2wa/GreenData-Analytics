# BigData Exam - Data Analysis Project

## 📊 Project Overview

This project involves comprehensive data analysis of plant sales data using Python, focusing on data cleaning, exploratory data analysis (EDA), and machine learning clustering techniques. The analysis includes three main datasets: Plant Facts, Account Analysis, and Plant Hierarchy data.

## 🗂️ Project Structure

```
BigData - Exam/
├── data/
│   ├── raw/
│   │   ├── Plant_DTS.xls
│   │   ├── AccountAnalysed.xlsx
│   │   └── Plant_Hearchy.xlsx
│   ├── cleaned/
│   │   └── cleaned_plant_fact.xlsx
│   └── enhanced/
│       ├── enhanced_plant_fact.xlsx
│       └── clustered_plant_fact.xlsx
├── notebooks/
│   └── data_analysis.py
└── powerbi/
    └── screenshots/
        ├── quantity_boxplot.png
        ├── quantity_dist_kde.png
        ├── correlation_heatmap.png
        ├── sales_vs_qty.png
        ├── price_vs_quantity.png
        ├── quantity_per_country.png
        └── kmeans_sales_quantity.png
```

## 🔧 Technologies Used

- **Python 3.x**
- **Libraries:**
  - pandas - Data manipulation and analysis
  - numpy - Numerical computing
  - matplotlib - Static plotting
  - seaborn - Statistical data visualization
  - scikit-learn - Machine learning tools
  - scipy - Scientific computing

## 📈 Dataset Description

### Source Files:

1. **Plant_DTS.xls** - Main plant facts data
2. **AccountAnalysed.xlsx** - Account information
3. **Plant_Hearchy.xlsx** - Product hierarchy data

### Key Variables:

- `quantity` - Product quantities sold
- `Sales_USD` - Sales amounts in USD
- `Price_USD` - Product prices in USD
- `COGS_USD` - Cost of goods sold
- `Date_Time` - Transaction timestamps
- `Account_id` - Account identifiers
- `Product_id` - Product identifiers
- `country_code` - Country information

## 🧹 Data Cleaning Process

### 1. Missing Value Handling

- Applied `dropna()` to remove incomplete records
- Ensured data integrity across all merged datasets

### 2. Data Type Standardization

- Converted `Date_Time` to proper datetime format
- Handled date parsing errors with `errors='coerce'`

### 3. Outlier Detection & Removal

- Used Interquartile Range (IQR) method for outlier detection
- Applied 1.5×IQR rule to identify and remove quantity outliers
- Formula: `Q1 - 1.5×IQR < quantity < Q3 + 1.5×IQR`

### 4. Feature Engineering

- Created `Sales_Per_Unit` feature: `Sales_USD / quantity`
- Applied one-hot encoding to `country_code` variables
- Standardized numerical features using StandardScaler

## 📊 Exploratory Data Analysis Results

### Descriptive Statistics Summary

| Metric    | Quantity | Sales_USD | Price_USD | COGS_USD  | Sales_Per_Unit |
| --------- | -------- | --------- | --------- | --------- | -------------- |
| **Count** | 2,440    | 2,440     | 2,440     | 2,440     | 2,440          |
| **Mean**  | 509.32   | 12,326.35 | 57.84     | 7,423.59  | 57.84          |
| **Std**   | 283.41   | 4,334.28  | 123.07    | 3,483.29  | 123.07         |
| **Min**   | 10.93    | 5,003.34  | 5.13      | 1,528.07  | 5.13           |
| **25%**   | 271.59   | 8,571.22  | 15.43     | 4,724.70  | 15.43          |
| **50%**   | 508.23   | 12,345.98 | 24.35     | 6,767.15  | 24.35          |
| **75%**   | 748.09   | 15,900.68 | 46.49     | 9,642.39  | 46.49          |
| **Max**   | 999.55   | 19,993.98 | 1,652.59  | 17,311.72 | 1,652.59       |

### Key Statistical Insights

- **Quantity Distribution:**

  - Mean: 509.32 units
  - Median: 508.23 units
  - Skewness: -0.01 (approximately normal distribution)
  - Range: 10.93 to 999.55 units

- **Sales Performance:**
  - Average sales per transaction: $12,326.35
  - Sales range: $5,003.34 to $19,993.98
  - Strong correlation with quantity sold

## 🤖 Machine Learning Analysis

### K-Means Clustering Implementation

**Objective:** Segment customers/transactions based on sales patterns

**Features Used:**

- `Sales_USD` (standardized)
- `quantity` (standardized)

**Model Configuration:**

- Algorithm: K-Means Clustering
- Random State: 42 (for reproducibility)
- Data Preprocessing: StandardScaler normalization

### Clustering Results

#### Optimal Cluster Analysis

Performed silhouette analysis for k=2 to k=10:

| k     | Silhouette Score |
| ----- | ---------------- |
| 2     | 0.358            |
| 3     | 0.383            |
| **4** | **0.411** ⭐     |
| 5     | 0.389            |
| 6     | 0.348            |
| 7     | 0.354            |
| 8     | 0.365            |
| 9     | 0.358            |
| 10    | 0.359            |

**Best Performance:** k=4 clusters with silhouette score of 0.411

### Model Evaluation

- **Primary Metric:** Silhouette Score
- **Best Score:** 0.411 (k=4 clusters)
- **Interpretation:** Moderate cluster separation, indicating distinct customer segments
- **Business Value:** Enables targeted marketing strategies for different customer groups

## 📈 Visualizations Generated

### 1. Distribution Analysis

- **Quantity Box Plot:** Outlier identification
- **Quantity Histogram + KDE:** Distribution shape analysis
- **Sales vs Quantity Scatter:** Relationship visualization

### 2. Correlation Analysis

- **Correlation Heatmap:** Inter-variable relationships
- Strong positive correlations identified between sales metrics

### 3. Clustering Visualizations

- **K-Means Scatter Plot:** Cluster separation visualization
- **Elbow Method:** Optimal cluster number determination
- **Silhouette Analysis:** Cluster quality assessment

### 4. Geographic Analysis

- **Sales by Country:** Regional performance comparison
- **Price vs Quantity by Country:** Market-specific patterns

## ⚠️ Technical Notes & Warnings

### Environment Warnings Encountered:

1. **Sklearn FutureWarning:** `n_init` parameter default change notification
2. **Memory Leak Warning:** KMeans Windows MKL compatibility issue
3. **Font Rendering:** Missing Unicode characters in matplotlib plots
4. **Layout Warnings:** Tight layout adjustments for complex plots

### Recommendations for Production:

- Set `n_init='auto'` explicitly in KMeans
- Configure `OMP_NUM_THREADS=10` for Windows environments
- Install comprehensive font packages for emoji rendering
- Implement custom layout management for complex visualizations

## 🎯 Key Findings & Business Insights

### 1. Data Quality

- Successfully processed 2,440 complete records
- Effective outlier removal improved data reliability
- Normal distribution in quantity suggests balanced sales patterns

### 2. Sales Patterns

- Strong linear relationship between quantity and sales revenue
- Average transaction value: $12,326.35
- Price per unit varies significantly (5.13 to 1,652.59 USD)

### 3. Customer Segmentation

- Identified 4 distinct customer clusters
- Moderate cluster separation suggests differentiated market segments
- Potential for targeted marketing strategies

### 4. Geographic Insights

- Country-specific sales patterns identified
- Regional variations in price sensitivity
- Opportunities for localized pricing strategies

## 🚀 Next Steps & Recommendations

### 1. Advanced Analytics

- Implement time series analysis for seasonal patterns
- Develop predictive models for sales forecasting
- Apply advanced clustering techniques (DBSCAN, Hierarchical)

### 2. Business Applications

- Create customer personas based on cluster analysis
- Develop targeted marketing campaigns
- Implement dynamic pricing strategies

### 3. Technical Improvements

- Automate data pipeline for real-time analysis
- Implement model monitoring and validation
- Create interactive dashboards for stakeholder insights

## 📋 How to Run the Analysis

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

### Execution

```bash
cd "C:\Users\Aime\Desktop\BigData - Exam"
python notebooks/data_analysis.py
```

### Output Locations

- Cleaned data: `data/cleaned/cleaned_plant_fact.xlsx`
- Enhanced data: `data/enhanced/enhanced_plant_fact.xlsx`
- Visualizations: `powerbi/screenshots/`
- Clustered results: `data/enhanced/clustered_plant_fact.xlsx`

## 📊 Performance Metrics Summary

- **Data Processing:** 2,440 records successfully processed
- **Model Performance:** Silhouette Score = 0.411
- **Feature Engineering:** 5 core features + country encoding
- **Visualization Output:** 7+ comprehensive charts generated
- **Processing Time:** Efficient execution with minimal computational overhead

---

**Project Completed:** August 2025  
**Analysis Framework:** Python-based Data Science Pipeline  
**Model Type:** Unsupervised Learning (K-Means Clustering)  
**Business Impact:** Customer segmentation and sales optimization insights
