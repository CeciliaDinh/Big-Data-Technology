# Set environment variables for Spark
import os
from tabs.home_page import show_data
from tabs.customer_churn import show_customer_churn
from tabs.general_dashboard import show_general_dashboard
from tabs.customer_segmentation import customer_segmentation

# Initialize Spark
import findspark
findspark.init()
from pyspark.sql import SparkSession
#Import Streamlit
import streamlit as st

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.1-bin-hadoop3"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StreamlitSparkApp") \
    .master("local[*]") \
    .getOrCreate()

def load_data(path):
     df = spark.read.csv(path, header=True, inferSchema=True)
     return df

# Sidebar configuration
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a side below.", ["Retail Data Overview", "Retail Analytics","Manage Customer Churn", "Customer Segmentation" ])


df_final = load_data("datasets\\final_df_churn_analyze.csv")
df = load_data("datasets\\Online-Retail-Cleaned.csv")
# Homepage
if page == "Retail Data Overview":
   show_data(df)
# Dashboard
elif page == "Retail Analytics":
    show_general_dashboard(df_final)
elif page == "Manage Customer Churn":
  show_customer_churn(df_final)
elif page == "Customer Segmentation":
  customer_segmentation(df_final)

# Stop Spark session
spark.stop()
