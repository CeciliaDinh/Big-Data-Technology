import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import TimestampType
from functools import reduce
import plotly.express as px
import pandas as pd

def show_data(df):
  st.title("Summary of Retail Transaction ")
  st.write("This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.")
  with st.expander("Online Retail Data Preview", expanded=False):
    st.dataframe(df.limit(5).toPandas())
  # count the number of rows and columns in the dataset
  st.subheader("📊 Data Dimensions")
  st.write(f"In the current time, this dataset contains : {len(df.columns)} features and {df.count()} online retails")
  # FE Get the max date of the dataset
  max_date = df.agg(max("InvoiceDate")).collect()[0][0]
  st.write(f"The last retail is : {max_date}")

  # Define the table data
  data = {
    "Schema ( Variables Name)": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
    "Role": ["ID", "ID", "Feature", "Feature", "Feature", "Feature", "Feature", "Feature"],
    "Type": ["Nominal", "Nominal", "Nominal", "Numeric", "Datetime", "Numeric", "Nominal", "Nominal"],
    "Description": [
        "a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation",
        "a 5-digit integral number uniquely assigned to each distinct product",
        "product name",
        "the quantities of each product (item) per transaction",
        "the day and time when each transaction was generated",
        "product price per unit",
        "a 5-digit integral number uniquely assigned to each customer",
        "the name of the country where each customer resides"
    ]
  }

  # Create a DataFrame
  df_descrip = pd.DataFrame(data)

  # Streamlit app title
  st.subheader("Key Dataset Metrics")
  # Display the table
  st.dataframe(df_descrip)



