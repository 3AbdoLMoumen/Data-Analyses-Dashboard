import pandas as pd
import numpy as np
import math

def Range(col):
    return np.max(col) - np.min(col)

def num_classes(R):
    return int(math.sqrt(R)) 

def numerical2classes(col):
    R = col.max() - col.min()
    numC = int(math.sqrt(R)) if R > 0 else 1
    r = R / numC if numC > 0 else 1
    start = col.min()
    bins = [start + i * r for i in range(numC)]
    bins.append(col.max() + 1e-8)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(numC)]
    return pd.cut(col, bins=bins, labels=labels, include_lowest=True)

def Frequency(col):

    return col.value_counts().reset_index().rename(columns={"index": col, col: "Frequency"})
