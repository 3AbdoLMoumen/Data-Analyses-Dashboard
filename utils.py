import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr

def Range(col):
    return np.max(col) - np.min(col)

def num_classes(R):
    return max(int(math.sqrt(R)), 2)  

def numerical2classes(col):
    R = Range(col)
    numC = num_classes(R)
    r = R / numC if numC > 0 else 1
    start = col.min()
    
    bins = [start + i * r for i in range(numC)]
    bins.append(col.max() + 1e-8)  
    
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(numC)]
    
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True)

def Frequency(col):
	freq = col.value_counts()
	return freq

def StatisticalParams(df, numerical_cols):
	return pd.DataFrame({
    "count": df[numerical_cols].count(),
    "mean": df[numerical_cols].mean(),
    "median": df[numerical_cols].median(),
    "mode": df[numerical_cols].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
    "min": df[numerical_cols].min(),
    "max": df[numerical_cols].max(),
    "range": df[numerical_cols].max() - df[numerical_cols].min(),
    "variance": df[numerical_cols].var(),
    "std_dev": df[numerical_cols].std(),
    "Q1": df[numerical_cols].quantile(0.25),
    "Q3": df[numerical_cols].quantile(0.75),
    "IQR": df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25)
})

def Correlations(x, y, alpha=0.05):
    r, p = pearsonr(x, y)

    if r > 0:
        relation = "positively correlated"
    elif r < 0:
        relation = "negatively correlated"
    else:
        relation = "not correlated"

    significance = "statistically significant" if p < alpha else "not statistically significant"

    return (
        f"The variables are {relation} "
        f"(Pearson r = {r:.3f}, p = {p:.3e}); "
        f"the correlation is {significance} at α = {alpha}."
    )
