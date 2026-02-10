import pandas as pd
import numpy as np
import math
import streamlit as st
import seaborn as sns
import plotly.express as px
from utils import *
from models import *
import seaborn as sns


st.set_page_config(
    page_title="Data Analytics Dashboard",
    page_icon="📊",
)


df = sns.load_dataset("tips").copy()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
dim = False
st.title("📊📈 Data Analytics Dashboard")
st.sidebar.header("Select Tool 🛠️")
tool = st.sidebar.selectbox("", ["Graphics", "Parameters", "Pair Plot"])

if tool == "Graphics":
    st.header("Graphical Representations 📊📉")
    st.sidebar.header("Select Variables")

    x_var = st.sidebar.selectbox("X variable", df.columns.tolist())
    x_is_numeric = False

    if x_var in numerical_cols:
        bin_x = st.sidebar.checkbox("Convert X to categorical (bin numeric)?", key="bin_x")
        if bin_x:
            df[f"{x_var}_Cat"] = numerical2classes(df[x_var])
            x_col = f"{x_var}_Cat"
        else:
            x_col = x_var
            x_is_numeric = True
    else:
        x_col = x_var

    y_var_input = st.sidebar.selectbox("Y variable (optional)", ["Frequency"] + df.columns.tolist())
    y_is_numeric = False

    if y_var_input == "Frequency":
        freq_df = df[x_col].value_counts().reset_index()
        freq_df.columns = [x_col, "Frequency"]
        y_col = "Frequency"
        plot_df = freq_df
    else:
        if y_var_input in numerical_cols:
            bin_y = st.sidebar.checkbox("Convert Y to categorical (bin numeric)?", key="bin_y")
            if bin_y:
                df[f"{y_var_input}_Cat"] = numerical2classes(df[y_var_input])
                y_col = f"{y_var_input}_Cat"
            else:
                y_col = y_var_input
                y_is_numeric = True
        else:
            y_col = y_var_input
        plot_df = df

    suggested_charts = []

    if x_is_numeric:
        if y_is_numeric:
            suggested_charts = ["Histogram", "Scatter"]
        elif y_var_input != "Frequency":
            suggested_charts = ["Box Plot"]
        else:
            suggested_charts = ["Histogram"]
    else:
        if y_is_numeric:
            suggested_charts = ["Bar", "Box Plot"]
        elif y_var_input != "Frequency":
            suggested_charts = ["Count Plot", "Pie Chart"]
        else:
            suggested_charts = ["Bar", "Pie Chart"]

    corr = None
    if x_is_numeric and y_is_numeric:
        corr = pearson_correlation(df[x_col], df[y_col])
    else:
        try:
            corr = spearman_correlation(encode(df[x_col]), encode(df[y_col]))
        except:
            pass
    if suggested_charts:
        chart_type = st.selectbox("Choose chart type", suggested_charts)

        if chart_type == "Histogram":
            fig = px.histogram(plot_df, x=x_col, y=y_col)

        elif chart_type == "Scatter":
            dim = st.sidebar.checkbox("3D")
            rad = st.sidebar.radio("Apply ML Model?", ["None", "Clustering-KMeans", "Anomaly Detection"])

            if not dim:
                X = plot_df[[x_col, y_col]]

                if rad == "Clustering-KMeans":
                    fig = px.scatter(plot_df, x=x_col, y=y_col, color=Cluster(X, st.sidebar.slider("n_Clusters", 2, 10, 3)))
                elif rad == "Anomaly Detection":
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=pd.Series(
                            AnomalyDetection(X, st.sidebar.slider("Contamination", 0.01, 0.2, 0.05))
                        ).map({1: "Normal", -1: "Anomaly"}),
                    )
                else:
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=st.sidebar.selectbox("Color by categorical column (optional)", [None] + categorical_cols),
                    )
            else:
                z_col = st.sidebar.selectbox("Z variable", numerical_cols)
                X = plot_df[[x_col, y_col, z_col]]

                if rad == "Clustering-KMeans":
                    fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=Cluster(X, st.sidebar.slider("n_Clusters", 2, 10, 3)))
                elif rad == "Anomaly Detection":
                    fig = px.scatter_3d(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=pd.Series(
                            AnomalyDetection(X, st.sidebar.slider("Contamination", 0.01, 0.2, 0.05))
                        ).map({1: "Normal", -1: "Anomaly"}),
                    )
                else:
                    fig = px.scatter_3d(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=st.sidebar.selectbox("Color by categorical column (optional)", [None] + categorical_cols),
                    )
                
        elif chart_type == "Bar":
            fig = px.bar(plot_df, x=x_col, y=y_col)

        elif chart_type == "Box Plot":
            fig = px.box(plot_df, x=x_col, y=y_col)

        elif chart_type == "Count Plot":
            fig = px.histogram(plot_df, x=x_col, color=y_col, barmode="group")

        elif chart_type == "Pie Chart":
            if y_col == "Frequency":
                fig = px.pie(plot_df, names=x_col)
            else:
                fig = px.pie(plot_df, names=y_col, facet_col=x_col)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No suitable charts for the selected combination.")

    if corr and not dim:
        st.header("Correlation Between X&Y Variables 📉📈")
        st.write(corr)

elif tool == "Parameters":
    st.header("📋📟 Statistical Parameters")
    st.write(StatisticalParams(df, numerical_cols))
elif tool=="Pair Plot":
    st.header("📶 Pair Plot")
    cat_choice = st.sidebar.selectbox("Color by categorical column (optional)", [None] + categorical_cols)
    pair_fig = sns.pairplot(df, hue=cat_choice) 
    st.pyplot(pair_fig)