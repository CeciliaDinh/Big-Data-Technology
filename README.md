# Big-Data-Technology course Project
# Customer-Segmentation-on-Retail-Data
A course project: [REPORT](https://drive.google.com/file/d/1mXwBqdlzoMzjBZlTc5sl11RYCCmfpV-e/view?usp=sharing)
## 📚 Table of Contents

1. [Team Members](#team-members)
2. [Overview](#overview)
3. [Objectives](#objectives)
4. [Tools & Technologies](#tools--technologies)
5. [Data Preprocessing](#data-preprocessing)
6. [Methodology](#methodology)
7. [Key Findings](#key-findings)
8. [Recommendations](#recommendations)

---

## Team Members

* Phạm Vũ Tuyết Anh
* Đinh Thị Thanh Hằng
* Lê Thị Tuyết My
* Đinh Vũ Ngọc Linh

## Overview

This project focuses on analyzing online retail transaction data using **data mining techniques** to uncover customer behavior patterns, segment customers, and support strategic marketing decisions. By leveraging **RFM analysis** and **K-Means clustering**, we provide actionable insights into customer value and engagement.

## Objectives

* Preprocess and clean retail transaction data from an international online store
* Derive **RFM (Recency, Frequency, Monetary)** features to assess customer value
* Apply **K-Means clustering** to group customers based on purchasing behavior
* Visualize and interpret customer segments to identify marketing opportunities
* Provide recommendations for customer retention and targeting strategies

## Tools & Technologies

* **Python**:

  * `Pandas` for data manipulation
  * `Matplotlib` & `Seaborn` for visualizations
  * `Scikit-learn` for clustering
* **Excel**: initial data exploration and profile summarization

## Data Preprocessing

The original dataset included over 500,000 rows of online retail transactions. Cleaning steps included:

* Removing missing values and canceled orders
* Filtering transactions with valid quantities and prices
* Aggregating customer-level data for RFM feature calculation

## Methodology

1. **RFM Analysis**:

   * **Recency**: How recently a customer made a purchase
   * **Frequency**: How often a customer makes a purchase
   * **Monetary**: How much a customer spends

2. **Clustering**:

   * Applied **K-Means algorithm** after normalizing RFM features
   * Used **Elbow Method** to determine the optimal number of clusters
   * Interpreted each cluster's characteristics to derive insights

## Key Findings

* **Cluster 0**: High-value, frequent, and recent customers → should be retained and rewarded
* **Cluster 1**: Low frequency, low spending customers → potential churn risks
* **Cluster 2**: High spenders with infrequent purchases → opportunity for re-engagement
* **Cluster 3**: Recent buyers with average frequency → target for upselling

## Recommendations

* Offer loyalty programs or discounts to Cluster 0 to maintain engagement
* Re-engage Cluster 2 with personalized promotions
* Monitor Cluster 1 for inactivity and test win-back strategies
* Nurture Cluster 3 with consistent follow-ups and cross-selling

