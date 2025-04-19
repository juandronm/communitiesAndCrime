import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("communities+and+crime/communities.data", header=None, na_values="?")
print(df.head())
print(df.isnull().sum())
print(df.isnull().sum().sum())

df_filled = df.fillna(df.median(numeric_only=True))
print(df_filled.head())
print(df_filled[0])