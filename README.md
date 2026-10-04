# 🛍️ Customer Segmentation Using RFM Analysis & Clustering

> **Unsupervised Machine Learning Project | Customer Segmentation | RFM Analysis**

---

## 📌 Project Overview

Customer segmentation is the process of dividing customers into meaningful groups based on their purchasing behaviour.

In this project, **RFM Analysis (Recency, Frequency, Monetary)** is combined with multiple **unsupervised machine learning clustering algorithms** to identify distinct customer segments from an online retail transaction dataset.

The project evaluates and compares:

* 🔵 **K-Means Clustering**
* 🟣 **Agglomerative Hierarchical Clustering**
* 🟢 **DBSCAN**

The primary objective is to identify meaningful customer groups and translate those groups into **actionable business strategies** for targeted marketing and customer retention.

---

## 🎯 Project Objectives

The main objectives of this project are:

* 🧹 Clean and preprocess the retail transaction data.
* 🔍 Perform exploratory data analysis (EDA).
* 👥 Create customer-level **RFM features**.
* 📊 Analyze distributions and identify outliers.
* 🔄 Apply appropriate transformations to reduce skewness.
* ⚖️ Scale the RFM features.
* 🤖 Apply multiple clustering algorithms.
* 📏 Evaluate clustering quality using the **Silhouette Score**.
* 🔁 Analyze the stability of the K-Means solution.
* 👤 Create meaningful customer personas.
* 💡 Develop business recommendations for each customer segment.
* 🚀 Save the final clustering model and scaler for future predictions.

---

# 📂 Dataset

The project uses the **Online Retail II** dataset containing transaction-level information from an online retail business.

Due to the large dataset size, the raw dataset is **not included directly in this GitHub repository**.

### 📥 Dataset Download

🔗 **[Download Online Retail II Dataset from Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)**

> 📌 Download the dataset from Kaggle and place the required dataset file in the project directory before running the notebook.

### 📊 Dataset Information

| Property                 | Details                    |
| ------------------------ | -------------------------- |
| 📌 Dataset               | Online Retail II           |
| 🌐 Source                | Kaggle                     |
| 🏢 Domain                | Online Retail / E-Commerce |
| 📦 Data Type             | Transactional Data         |
| 🎯 Project Task          | Customer Segmentation      |
| 🧠 Machine Learning Type | Unsupervised Learning      |
| 📊 Main Technique        | RFM Analysis + Clustering  |

### 📋 Dataset Features

| Feature       | Description                  |
| ------------- | ---------------------------- |
| `Invoice`     | Invoice / transaction number |
| `StockCode`   | Product code                 |
| `Description` | Product description          |
| `Quantity`    | Number of products purchased |
| `InvoiceDate` | Date and time of transaction |
| `Price`       | Unit price of the product    |
| `Customer ID` | Unique customer identifier   |
| `Country`     | Customer's country           |

> ⚠️ **Note:** The dataset is intentionally excluded from this repository because of its large file size.

---

# 🔎 1. Exploratory Data Analysis

The dataset was first explored to understand its structure, distributions, transaction behaviour, and customer purchasing patterns.

## 🧹 Data Filtering & Cleaning

The following preprocessing steps were performed:

1. 🇬🇧 Filtered the dataset to **United Kingdom** transactions.
2. ❌ Removed records where `Customer ID` was missing.
3. ❌ Removed transactions where `Quantity <= 0`.
4. ❌ Removed transactions where `Price <= 0`.
5. 💰 Created a new `TotalPrice` feature.

### 💵 Total Price

The total value of each transaction was calculated as:

```python
TotalPrice = Quantity × Price
```

---

## 📊 EDA Performed

The project includes exploratory analyses such as:

* 📈 Histograms
* 🌍 Country-level transaction analysis
* 📅 Monthly transaction analysis
* 👥 Customer-level purchase analysis
* 🛒 Analysis of frequent buyers
* 📦 Distribution analysis of transaction variables
* 🔍 Outlier analysis

### 📅 Seasonal Transaction Behaviour

The analysis identified noticeable seasonal patterns in transaction activity.

Key observations include:

* 📈 Transaction activity increases significantly during **October and November**.
* 🔥 The highest transaction activity occurs around **November 2011**.
* 🛍️ Similar seasonal increases are visible during the previous year's holiday period.
* 📉 Transaction activity decreases after the year-end peak.

This indicates a strong **holiday / year-end shopping effect** within the transaction data.

---

# 👥 2. RFM Feature Engineering

To perform customer segmentation, transaction-level data was transformed into a customer-level dataset using **RFM Analysis**.

## 🧠 What is RFM?

RFM represents three important aspects of customer purchasing behaviour:

### 🔵 Recency

Measures how recently a customer made a purchase.

```text
Recency = Reference Date − Most Recent Purchase Date
```

A **lower Recency value** indicates a more recently active customer.

---

### 🟣 Frequency

Measures how frequently a customer makes purchases.

In this project:

```text
Frequency = Number of Unique Invoices
```

A **higher Frequency** indicates more frequent purchasing behaviour.

---

### 🟢 Monetary

Measures the total amount spent by the customer.

```text
Monetary = Sum of TotalPrice
```

A **higher Monetary value** indicates greater customer value.

---

## 📅 Reference Date

The RFM analysis uses:

```text
Reference Date = 2011-12-31
```

The resulting customer-level RFM dataset contains:

```text
5,350 Customers
```

with the following core features:

```text
Recency
Frequency
Monetary
```

---

# 🧹 3. RFM Preprocessing

## 📦 Outlier Handling

RFM variables can contain extreme values, particularly **Frequency** and **Monetary**.

An upper limit based on:

```text
Upper Limit = Q3 + 3 × IQR
```

was used to cap extreme observations.

This reduces the influence of unusually large customer values while retaining the customers in the dataset.

---

## 🔄 Log Transformation

Frequency and Monetary showed right-skewed distributions.

Therefore, `log1p()` transformation was applied:

```python
Frequency = log1p(Frequency)
Monetary = log1p(Monetary)
```

Recency was kept unchanged.

### 📌 Why was log transformation used?

Log transformation helps reduce the influence of extreme values and produces distributions that are more suitable for distance-based clustering algorithms.

---

# ⚖️ 4. Feature Scaling

Since clustering algorithms rely heavily on distance calculations, the RFM features were standardized using:

```python
StandardScaler()
```

The final clustering features were:

```text
Recency
Frequency_log
Monetary_log
```

Standardization ensures that features with different numerical scales do not disproportionately influence the clustering process.

---

# 🤖 5. K-Means Clustering

K-Means was used as the primary clustering algorithm.

## 🔍 Selecting the Number of Clusters

The **Elbow Method** and **Silhouette Score** were used to determine a suitable number of clusters.

Although the highest Silhouette Score occurred at `k = 2`, the elbow curve showed a clear bend around:

```text
k = 3
```

Therefore:

```text
Optimal K = 3
```

was selected because it provided a good balance between clustering quality and meaningful business segmentation.

---

## ⚙️ Final K-Means Configuration

```python
KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=20,
    max_iter=500,
    random_state=42
)
```

### 📊 K-Means Performance

```text
Silhouette Score = 0.409
```

K-Means produced **three customer segments**.

---

# 🌳 6. Agglomerative Hierarchical Clustering

Agglomerative Hierarchical Clustering was applied to compare its segmentation performance with K-Means.

## 📌 Dendrogram

A dendrogram using **Ward linkage** was created to investigate the hierarchical structure of the customer data.

The analysis supported a solution containing:

```text
3 Clusters
```

---

## 🔬 Linkage Comparison

Different linkage methods were evaluated:

* `ward`
* `complete`
* `average`

The best-performing linkage was:

```text
Ward
```

### 📊 Performance

```text
Silhouette Score = 0.386
```

The hierarchical clustering solution largely matched the K-Means segmentation, although some customers were assigned differently around cluster boundaries.

---

# 🌐 7. DBSCAN Clustering

DBSCAN was used to identify density-based customer groups and unusual customer behaviour.

Unlike K-Means and Agglomerative Clustering, DBSCAN can explicitly identify **noise / outlier customers**.

## 🎛️ Hyperparameter Tuning

An initial k-NN distance analysis suggested an epsilon around:

```text
ε ≈ 0.25
```

Further tuning was performed using:

```text
eps = [0.3, 0.5, 0.7, 1.0, 1.5]

min_samples = [3, 5, 8, 10]
```

The final DBSCAN configuration was:

```python
DBSCAN(
    eps=0.5,
    min_samples=5,
    metric="euclidean"
)
```

### 📊 DBSCAN Performance

* 🔢 Number of clusters: **2**
* 📏 Silhouette Score: **0.333**
* ⚠️ Noise: approximately **0.49%**

DBSCAN was useful for identifying customers whose RFM behaviour did not fit neatly into the main customer groups.

---

# 📊 8. Model Comparison

The clustering algorithms were compared using the **Silhouette Score**.

| Algorithm        | Configuration                  | Clusters | Silhouette Score | Noise |
| ---------------- | ------------------------------ | -------: | ---------------: | ----: |
| 🔵 K-Means       | `k=3, n_init=20, max_iter=500` |        3 |        **0.409** |    0% |
| 🟣 Agglomerative | `n_clusters=3, Ward`           |        3 |            0.386 |    0% |
| 🟢 DBSCAN        | `eps=0.5, min_samples=5`       |        2 |            0.333 | 0.49% |

## 🏆 Best Model

Based on:

* 📈 Highest Silhouette Score
* 🎯 Meaningful customer segmentation
* 🔁 Strong stability
* 💼 Business interpretability

**K-Means Clustering was selected as the final model.**

---

# 🔁 9. K-Means Stability Analysis

To verify that the clustering solution was not highly dependent on a single random state, K-Means was trained using multiple random states:

```text
0
7
21
42
99
```

### 📊 Stability Result

```text
Mean Silhouette Score = 0.4096
Standard Deviation   = 0.0002
```

Therefore:

```text
0.4096 ± 0.0002
```

The extremely small standard deviation indicates that the K-Means clustering solution is highly stable across different runs.

---

# 👤 10. Customer Segmentation & Personas

The final K-Means solution contains **3 customer segments**.

The segments were interpreted according to their RFM behaviour.

---

## 🏆 1. Champions

These customers demonstrate relatively strong purchasing activity and high customer value.

### 💡 Recommended Strategy

* ⭐ Introduce an exclusive loyalty program.
* 🎁 Provide premium rewards.
* 🚀 Give early access to new products and sales.
* 💎 Provide special benefits to maintain loyalty.

---

## 💙 2. Loyal Customers

These customers show regular purchasing behaviour and provide consistent business value, although they are not the highest-value group.

### 💡 Recommended Strategy

* 🎟️ Offer personalized discounts.
* 🎁 Provide reward points.
* 🔁 Encourage repeat purchases.
* 📈 Use targeted offers to increase purchase frequency.

---

## ⚠️ 3. At-Risk / Hibernating Customers

These customers have relatively high Recency, indicating that they have not purchased recently.

They represent customers who may be at risk of becoming inactive.

### 💡 Recommended Strategy

* 📧 Launch targeted re-engagement campaigns.
* 🎟️ Provide limited-time discount coupons.
* ⏳ Create offers with short expiration periods.
* 🔄 Encourage customers to return and purchase again.

---

# 💼 11. Business Recommendations

The RFM segmentation can help a marketing team move away from a **one-size-fits-all marketing strategy**.

Instead, campaigns can be customized according to customer behaviour.

| Customer Segment     | Business Goal               | Suggested Action                              |
| -------------------- | --------------------------- | --------------------------------------------- |
| 🏆 Champions         | Retain high-value customers | Loyalty programs & premium rewards            |
| 💙 Loyal Customers   | Increase customer value     | Personalized discounts & reward points        |
| ⚠️ At-Risk Customers | Reactivate customers        | Re-engagement campaigns & limited-time offers |

---

# 📈 12. Future Improvements

RFM provides a strong foundation for customer segmentation, but additional information could improve segmentation quality.

Potential future features include:

### 👤 Customer Information

* Age
* Gender
* Location

### 🛍️ Product Behaviour

* Product categories
* Product preferences
* Frequently purchased products

### 💰 Purchase Behaviour

* Average Order Value
* Discount usage
* Coupon usage

### 📱 Customer Engagement

* Website visits
* Mobile app activity
* Click behaviour
* Browsing behaviour

### 🔄 Customer Activity

* Returns
* Cancellations
* Customer complaints
* Customer support interactions

### ⭐ Customer Satisfaction

* Ratings
* Reviews
* Customer feedback

Combining these variables with RFM could provide a more comprehensive view of customer behaviour.

---

# 🚀 13. Deployment & Prediction Pipeline

The final clustering workflow was prepared for future customer segmentation.

The following objects were saved:

```text
rfm_scaler.pkl
customer_segmentation_model.pkl
```

The prediction pipeline follows:

```text
New Customer RFM Data
        ↓
Log1p Transformation
        ↓
Standard Scaling
        ↓
K-Means Model
        ↓
Cluster Assignment
        ↓
Customer Persona
```

This allows new customers to be assigned to an existing customer segment using the trained pipeline.

---

# 🛠️ Technologies & Libraries

## 🐍 Programming Language

* Python

## 📚 Libraries

* 🐼 **Pandas**
* 🔢 **NumPy**
* 📊 **Matplotlib**
* 🎨 **Seaborn**
* 🤖 **Scikit-learn**
* 💾 **Joblib**
* 🌳 **SciPy**

## 🤖 Machine Learning Techniques

* RFM Analysis
* K-Means Clustering
* Agglomerative Hierarchical Clustering
* DBSCAN
* Standardization
* Log Transformation
* Outlier Capping
* Silhouette Score
* Elbow Method
* Dendrogram Analysis
* Hyperparameter Tuning
* Model Stability Analysis

---

# 📁 Project Structure

The dataset is **not included** in this repository because of its large file size.

After downloading the dataset from Kaggle, the project can be organized as:

```text
Customer-Segmentation/
│
├── 📓 Customer_Segmentation.ipynb
│
├── 📊 online_retail_II.xlsx
│
├── 📦 rfm_scaler.pkl
│
├── 🤖 customer_segmentation_model.pkl
│
└── 📄 README.md
```

### 📥 Dataset

Download the dataset separately from:

🔗 **[Kaggle — Online Retail II (UCI)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)**

---

# 🧠 Key Learning Outcomes

Through this project, the following concepts were practically implemented:

* 🔍 Exploratory Data Analysis
* 🧹 Data Cleaning
* 🛠️ Feature Engineering
* 👥 Customer-Level Aggregation
* 📊 RFM Analysis
* 📦 Outlier Handling
* 🔄 Log Transformation
* ⚖️ Feature Scaling
* 🤖 Unsupervised Machine Learning
* 📍 K-Means Clustering
* 🌳 Hierarchical Clustering
* 🌐 DBSCAN
* 📏 Silhouette Score
* 📉 Elbow Method
* 🌲 Dendrogram Analysis
* 🔁 Model Stability Analysis
* 💼 Business Interpretation
* 🚀 Model Deployment Preparation

---

# 🏁 Final Conclusion

This project demonstrates how **RFM-based customer segmentation** can be combined with unsupervised machine learning to identify meaningful customer groups from transactional data.

Among the three clustering approaches evaluated, **K-Means achieved the highest Silhouette Score of 0.409** and demonstrated excellent stability with a mean Silhouette Score of:

```text
0.4096 ± 0.0002
```

Therefore, **K-Means Clustering was selected as the final segmentation approach**.

The resulting customer segments can help businesses develop **targeted marketing strategies**, improve customer retention, increase customer value, and reduce customer inactivity.

> 💡 **Final Insight:**
> Data-driven customer segmentation allows businesses to understand **who their customers are, how they behave, and how different customer groups should be approached differently.**

---

# ⭐ Project Highlights

```text
📊 1M+ Transaction Records
        ↓
🧹 Data Cleaning & EDA
        ↓
🇬🇧 UK Customer Transactions
        ↓
👥 5,350 Customers
        ↓
📌 RFM Feature Engineering
        ↓
📦 Outlier Handling
        ↓
🔄 Log Transformation
        ↓
⚖️ Standard Scaling
        ↓
🤖 K-Means + Hierarchical + DBSCAN
        ↓
📏 Model Evaluation
        ↓
🏆 K-Means Selected
        ↓
👤 3 Customer Segments
        ↓
💼 Targeted Marketing Strategies
        ↓
🚀 Deployment-Ready Pipeline
```
---


Feel free to **⭐ star the repository** and explore the notebook to understand the complete customer segmentation workflow.
