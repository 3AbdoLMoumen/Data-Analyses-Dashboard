import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def Cluster(X, clusters=2):
    km = KMeans(n_clusters=clusters, random_state=0, n_init="auto")
    labels = km.fit_predict(X)  
    return labels


def AnomalyDetection(X, contamination=0.05, random_state=0):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)
    return labels
