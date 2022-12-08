import statistics as stat
import pandas as pd

def mathMean(data, column):
    dataList = list(data[column].values)
    return stat.mean(dataList)

def stndError(data, column):
    dataList = list(data[column].values)
    return stat.stdev(dataList)

def minimum(data, column):
    dataList = list(data[column].values)
    return min(dataList)

def maximum(data, column):
    dataList = list(data[column].values)
    return max(dataList)

def quantile(data:pd.DataFrame,axis:int, q:int):
    return data.quantile(q, axis=0)