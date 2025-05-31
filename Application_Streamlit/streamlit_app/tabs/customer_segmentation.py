from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
import plotly.express as px
import streamlit as st
import pandas as pd
from pyspark.ml.feature import StandardScaler

def load_model():
    try:
        model = KMeans(featuresCol="scaled_features", predictionCol="cluster",k= 3, seed =1)
        st.write("KMeans Algorithm is applied to segment customers into clusters")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

def load_cluster_data(df):
    try:
        rfm = df.select("CustomerID", "Recency", "Frequency", "Monetary")
        assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
        data_vector = assembler.transform(rfm)


        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
        scaler_model = scaler.fit(data_vector)
        scaled_data = scaler_model.transform(data_vector)
        return scaled_data
    except Exception as e:
        st.error(f"Failed to load or transform data. Error: {e}")
        return None

def customer_segmentation(df):
    st.title("Customer Segmentation")

    # Load model and data
    model = load_model()
    data = load_cluster_data(df)

    if model is None or data is None:
        return

    # Predict clusters
    model = model.fit(data)
    cluster_data = model.transform(data)

    # Evaluate clustering
    evaluator = ClusteringEvaluator(featuresCol="scaled_features", predictionCol="cluster", metricName="silhouette")
    sil_score = evaluator.evaluate(cluster_data)

    st.subheader("Clustered Data")
    cols_to_drop = [col for col in ["features", "scaled_features"] if col in cluster_data.columns]
    df_unique = cluster_data.drop(*cols_to_drop).distinct()
    st.dataframe(df_unique.toPandas())

    st.write(f"**Silhouette Score:** {sil_score:.3f}")

    # # Get number of clusters dynamically
    n_clusters = len(model.clusterCenters())
    cluster_labels = [f"Cluster {i+1}" for i in range(n_clusters)]
    cluster_dfs = [cluster_data.filter(F.col("cluster") == i) for i in range(n_clusters)]
    cluster_counts = [c.count() for c in cluster_dfs]
    # # Plot cluster distribution as pie chart
    # st.subheader("Cluster Distribution")
    # fig = px.pie(
    # names=cluster_labels,
    # values=cluster_counts,
    # title="Customer Segmentation by Cluster",
    # color_discrete_sequence=px.colors.qualitative.Set2
    # )
    # fig.update_traces(textposition='inside', textinfo='percent+label')
    # st.plotly_chart(fig)

    # Cluster interpretation (optional: adapt based on actual model insight)
    st.subheader("Cluster Interpretation")

    # Adjust the interpretations according to the actual number of clusters if needed
    cluster_info = {
        "Cluster": [0,1,2],
        "Label": [
            "Active Medium-Value Customers",
            "Ultra VIP Customer",
            "Churned Low-Value Customers"
        ][:n_clusters],
        "Key Traits": [
            "Regular, recent, moderate to high spending",
            "Bought recently, frequently, and very high value",
            "Long inactive, rare purchases, low value"
        ][:n_clusters]
    }

    cluster_df = pd.DataFrame(cluster_info)
    st.table(cluster_df)

    summary_df = cluster_data.groupBy("cluster").agg(
    F.count("*").alias("CustomerCount"),
    F.sum("Recency").alias("days_since_last_purchase"),
    F.sum("Frequency").alias("total_purchases"),
    F.sum("Monetary").alias("total_sales")
    )

    # Convert to Pandas for plotting
    summary_pd = summary_df.toPandas()

    # Create labels
    summary_pd["ClusterLabel"] = summary_pd["cluster"].apply(lambda x: f"Cluster {x}")

    st.title("Customer Segmentation Analysis")
    col1, col2 = st.columns(2)
    with col1:
      # Pie Chart 1: Customer Count
      fig1 = px.pie(summary_pd, names="ClusterLabel", values="CustomerCount", title="Customer Segmentation by Cluster", color_discrete_sequence=px.colors.qualitative.Set2)
      st.plotly_chart(fig1)
    with col2:
      # Pie Chart 2: Sales Contribution
      fig2 = px.pie(summary_pd, names="ClusterLabel", values="total_sales", title="Sales Contribution by Cluster", color_discrete_sequence=px.colors.qualitative.Set3)
      st.plotly_chart(fig2)
    col3, col4 = st.columns(2)
    with col3:
      # Pie Chart 3: Recency
      fig3 = px.pie(summary_pd, names="ClusterLabel", values="days_since_last_purchase", title="Recency by Cluster", color_discrete_sequence=px.colors.qualitative.Pastel)
      st.plotly_chart(fig3)
    with col4:
      # Pie Chart 4: Frequency
      fig4 = px.pie(summary_pd, names="ClusterLabel", values="total_purchases", title="Frequency by Cluster", color_discrete_sequence=px.colors.qualitative.Bold  )
      st.plotly_chart(fig4)
