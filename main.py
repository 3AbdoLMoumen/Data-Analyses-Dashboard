import pandas as pd
import numpy as np
import math
import streamlit as st
import seaborn as sns
import plotly.express as px
from utils import *

# --- Load Data ---
df = sns.load_dataset("tips").copy()

st.title("Interactive Data Dashboard")
st.sidebar.header("Select Variables")

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# --- X Variable ---
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

# --- Y Variable ---
y_var_input = st.sidebar.selectbox("Y variable (optional)", ["Frequency"] + df.columns.tolist())
y_is_numeric = False

if y_var_input == "Frequency":
    freq_df = df[x_col].value_counts().reset_index()
    freq_df.columns = [x_col, 'Frequency']
    y_col = 'Frequency'
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

# --- Suggest Charts ---
suggested_charts = []

if x_is_numeric:
    if y_is_numeric:
        suggested_charts = ["Histogram", "Scatter", "Line"]
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

# --- Plot ---
if suggested_charts:
    chart_type = st.selectbox("Choose chart type", suggested_charts)

    if chart_type == "Histogram":
        fig = px.histogram(plot_df, x=x_col, y=y_col)
    elif chart_type == "Scatter":
        fig = px.scatter(plot_df, x=x_col, y=y_col, trendline="ols")
    elif chart_type == "Line":
        fig = px.line(plot_df, x=x_col, y=y_col)
    elif chart_type == "Bar":
        fig = px.bar(plot_df, x=x_col, y=y_col)
    elif chart_type == "Box Plot":
        fig = px.box(plot_df, x=x_col, y=y_col)
    elif chart_type == "Count Plot":
        fig = px.histogram(plot_df, x=x_col, color=y_col, barmode="group")
    elif chart_type == "Pie Chart":
        fig = px.pie(plot_df, names=x_col, values=y_col)
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No suitable charts for the selected combination.")

st.write(
	StatisticalParams(df, numerical_cols)
)
