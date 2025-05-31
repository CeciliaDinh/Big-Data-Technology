import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

from pyspark.sql.functions import (
    avg, col, countDistinct, desc, when, sum as _sum,
    percentile_approx
)

def render_bar_chart(df, column, top_type, title):
    """Render a horizontal bar chart for top or lowest N records."""
    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()
    ascending = top_type.lower() == "lowest n"
    df_sorted = df.sort_values(by=column, ascending=ascending).head(5)
    fig = px.bar(
        df_sorted,
        x=column,
        y='CustomerID',
        orientation='h',
        hover_data=['CustomerID'],
        title=f"{title} - Top 5 {top_type}"
    )
    st.plotly_chart(fig)

def show_rfm_averages(df):
    st.title("Analyze Customer Churn based on RMF")
    ave_df = df.groupBy("Churn").agg(
        avg("Recency").alias("Avg_Recency"),
        avg("Frequency").alias("Avg_Frequency"),
        avg("Monetary").alias("Avg_Monetary"),
        countDistinct("CustomerID").alias("Num_Customers")
    )
    with st.expander("Average Recency, Frequency, Monetary by Churn", expanded=False):
        st.dataframe(ave_df)

def show_product_analysis(df):
    st.subheader("Product Aspect")
    recency_threshold = df.select(percentile_approx("Recency", 0.75).alias("recency_75th")).collect()[0]["recency_75th"]
    high_recency_customers = df.filter(col("Recency") >= recency_threshold).select("CustomerID")
    high_recency_txns = high_recency_customers.join(df, on="CustomerID", how="inner")

    product_counts = high_recency_txns.groupBy("StockCode").count().orderBy(desc("count"))
    product_pd = product_counts.toPandas()

    option = st.selectbox("Select view:", ["Top N", "Lowest N"])
    n = st.slider("How many stocks to display:", min_value=1, max_value=10, value=5)

    ascending = option == "Lowest N"
    sorted_data = product_pd.sort_values(by="count", ascending=ascending).head(n)

    chart = alt.Chart(sorted_data).mark_bar().encode(
        x='count',
        y=alt.Y('StockCode', sort='-x'),
        tooltip=['StockCode', 'count']
    ).properties(
        width=600,
        height=400,
        title=f"{option} Stocks Code by high Recency"
    )

    st.altair_chart(chart)

def show_people_analysis(df):
    st.subheader("People Aspect")
    st.write("Which churned customers had high past value? Can they be targeted for reactivation?")
    high_value_threshold = df.approxQuantile("Monetary", [0.75], 0.01)[0]
    high_value_churned = df.filter(
        (col("Churn") == 1) & (col("Monetary") >= high_value_threshold)
    ).select("CustomerID", "Recency", "Frequency", "Monetary")

    st.dataframe(high_value_churned)

    revenue_df = df.withColumn("Revenue", col("Quantity") * col("UnitPrice")) \
        .groupBy("Country", "Churn").agg(
            _sum("Revenue").alias("Total_Revenue"),
            countDistinct("CustomerID").alias("Num_Customers")
        ).withColumn("Revenue_per_Customer", col("Total_Revenue") / col("Num_Customers"))

    revenue_pd = revenue_df.select("Country", "Churn", "Revenue_per_Customer").toPandas()
    pivot_df = revenue_pd.pivot(index="Country", columns="Churn", values="Revenue_per_Customer")
    pivot_df = pivot_df.rename(columns={0: "Active", 1: "Churned"})
    pivot_df["Total_Avg"] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values("Total_Avg", ascending=False)

    df_plot = pivot_df.reset_index().melt(
        id_vars='Country',
        value_vars=['Active', 'Churned'],
        var_name='Churn Status',
        value_name='Revenue per Customer'
    )

    fig = px.bar(
        df_plot,
        x='Country',
        y='Revenue per Customer',
        color='Churn Status',
        barmode='group',
        title="Revenue per Customer by Country and Churn Status",
        height=600
    )
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Revenue per Customer",
        xaxis_tickangle=-45,
        legend_title_text="Churn Status",
        margin=dict(t=50, b=100)
    )
    st.plotly_chart(fig)


    churned_df = df.filter(col("Churn") == 1)
    top_products = churned_df.groupBy("StockCode", "Description").agg(
        _sum("Quantity").alias("TotalQuantity")
    ).orderBy(desc("TotalQuantity")).limit(3)

    top_pd = top_products.toPandas()

    fig = px.bar(
        top_pd,
        x="TotalQuantity",
        y="Description",
        orientation='h',
        labels={"TotalQuantity": "Total Quantity"},
        title="Products Most Bought by Churned Customers"
    )
    st.plotly_chart(fig)

def show_place_analysis(df):
    st.subheader("Place Aspect")
    churn_by_country = df.groupBy("Country").agg(
        countDistinct(when(col("Churn") == 1, col("CustomerID"))).alias("Churned_Customers"),
        countDistinct(when(col("Churn") == 0, col("CustomerID"))).alias("Active_Customers")
    ).withColumn(
        "Churn_Rate",
        col("Churned_Customers") / (col("Churned_Customers") + col("Active_Customers"))
    )

    churn_pd = churn_by_country.orderBy(col("Churn_Rate")).toPandas()
    data_sorted = churn_pd.sort_values(by="Churn_Rate", ascending=False)

    fig = px.bar(
        data_sorted,
        x="Churn_Rate",
        y="Country",
        orientation='h',
        color="Churn_Rate",
        color_continuous_scale='Reds',
        labels={"Churn_Rate": "Churn Rate"},
        title="Churn Rate by Country"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)


def show_customer_churn(df):
    show_rfm_averages(df)
    st.markdown("### The below RMF analysis follows the 3Ps-G framework")
    show_product_analysis(df)
    show_people_analysis(df)
    show_place_analysis(df)
