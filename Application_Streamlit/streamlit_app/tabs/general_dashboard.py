from pyspark.sql.functions import (
    col, sum as _sum, month, year, lag, round, countDistinct
)
from pyspark.sql.window import Window
import streamlit as st
import plotly.express as px
from pyspark.sql import functions as F

def show_general_dashboard(df):
    st.title("Overall Analysis By All Factors")
      # --- Revenue by Country ---
    revenue_by_country_df = df.withColumn("Revenue", col("Quantity") * col("UnitPrice")) \
        .groupBy("Country") \
        .agg(_sum("Revenue").alias("Total_Revenue")) \
        .orderBy(col("Total_Revenue").desc())

    revenue_pd = revenue_by_country_df.toPandas()
    fig = px.bar(
        revenue_pd,
        x="Country",
        y="Total_Revenue",
        labels={"Total_Revenue": "Total Revenue"},
        title="Revenue per Country",
        color_discrete_sequence=["skyblue"]
      )
    fig.update_layout(xaxis_tickangle=-90, yaxis_title="Total Revenue", xaxis_title="Country")
    st.plotly_chart(fig, use_container_width=True)  # or st.altair_chart()

    # --- Top Countries Revenue Chart ---
    st.subheader("Top Countries in Revenue")
    top_n = st.slider('Select Top N Countries', min_value=1, max_value=10, value=5)
    sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
    ascending = sort_order == "Ascending"

    top_countries_df = revenue_by_country_df.orderBy(col("Total_Revenue").asc() if ascending else col("Total_Revenue").desc()) \
        .limit(top_n)
    top_pd = top_countries_df.toPandas()
    sorted_countries = top_pd.sort_values(by='Total_Revenue', ascending=ascending)["Country"].tolist()

    fig = px.bar(
        top_pd.sort_values(by='Total_Revenue', ascending=ascending),
        x='Total_Revenue',
        y='Country',
        orientation='h',
        text='Total_Revenue',
        title=f"Top {top_n} Countries by Revenue ({sort_order})",
        category_orders={"Country": sorted_countries}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig)


    # Extract year from InvoiceDate
    df_with_year = df.withColumn("Year", year("InvoiceDate"))

    # Revenue per year and country
    revenue_per_year = df_with_year.groupBy("Country", "Year") \
      .agg(F.sum("Monetary").alias("YearlyRevenue"))

    # Define window for YoY calculation
    window_spec = Window.partitionBy("Country").orderBy("Year")

    # Calculate YoY Growth
    revenue_growth = revenue_per_year.withColumn(
      "PreviousRevenue", lag("YearlyRevenue").over(window_spec)
      ).withColumn(
      "YoY_Growth", round(((col("YearlyRevenue") - col("PreviousRevenue")) / col("PreviousRevenue")) * 100, 2)
      ).orderBy("Country", "Year")

    # Display in Streamlit
    st.subheader("YoY Revenue Growth per Country")
    st.dataframe(revenue_growth.toPandas().head(5))

    # --- Monthly Revenue Trend for Top Countries ---
    top_countries = [row["Country"] for row in top_countries_df.collect()]
    filtered_df = df.filter(col("Country").isin(top_countries))

    monthly_trend_df = filtered_df.withColumn("Revenue", col("Quantity") * col("UnitPrice")) \
        .withColumn("Year", year("InvoiceDate")) \
        .withColumn("Month", month("InvoiceDate")) \
        .groupBy("Year", "Month", "Country") \
        .agg(_sum("Revenue").alias("Total_Revenue")) \
        .orderBy("Year", "Month")

    monthly_pd = monthly_trend_df.toPandas()
    monthly_pd["YearMonth"] = monthly_pd["Year"].astype(str) + "-" + monthly_pd["Month"].astype(str).str.zfill(2)
    monthly_pd = monthly_pd.sort_values(["Year", "Month"])

    fig = px.line(
        monthly_pd,
        x="YearMonth",
        y="Total_Revenue",
        color="Country",
        markers=True,
        title="Monthly Revenue Trend of Top Countries"
    )
    fig.update_layout(xaxis_tickangle=45, template="plotly_white")
    st.plotly_chart(fig)

    # --- Top Products by Quantity ---
    st.subheader("Top Products by Quantity")
    top_n_products = st.slider('Top N Products by Quantity', 1, 10, 5, key="top_n_products")
    sort_order_products = st.selectbox("Product Sort Order", ["Descending", "Ascending"], key="sort_order_products")
    ascending_products = sort_order_products == "Ascending"

    top_products_df = df.groupBy("Description") \
        .agg(_sum("Quantity").alias("TotalQuantity")) \
        .orderBy(col("TotalQuantity").asc() if ascending_products else col("TotalQuantity").desc()) \
        .limit(top_n_products)

    product_pd = top_products_df.toPandas()
    fig = px.bar(
        product_pd.sort_values(by="TotalQuantity", ascending=False),
        x="TotalQuantity",
        y="Description",
        orientation='h',
        text="TotalQuantity",
        title=f"Top {top_n_products} Products by Quantity ({sort_order_products})",
        category_orders={"Description": product_pd.sort_values(by='TotalQuantity', ascending=False)["Description"].tolist()}

    )
    st.plotly_chart(fig)

    # --- Top Products by YoY Revenue Growth ---
    st.subheader("Products with Highest YoY Revenue Growth")
    product_df = df.withColumn("Year", year("InvoiceDate"))

    product_year_revenue = product_df.groupBy("StockCode", "Description", "Year") \
        .agg(_sum(col("Quantity") * col("UnitPrice")).alias("YearlyRevenue"))

    window_spec = Window.partitionBy("StockCode").orderBy("Year")
    growth_df = product_year_revenue.withColumn(
        "PreviousYearRevenue", lag("YearlyRevenue").over(window_spec)
    ).withColumn(
        "YoY_Growth", ((col("YearlyRevenue") - col("PreviousYearRevenue")) / col("PreviousYearRevenue")) * 100
    ).filter(col("PreviousYearRevenue").isNotNull())

    top_growth_products = growth_df.orderBy(col("YoY_Growth").desc()).limit(5)
    st.dataframe(top_growth_products.toPandas())


