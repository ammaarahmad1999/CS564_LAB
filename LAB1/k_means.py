# -*- coding: utf-8 -*-
"""K_Means.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VNWdBMpUjv_yXYhv1BCy35mVn6_IhD0K
"""

import pandas as pd
import numpy as np

import math

# Commented out IPython magic to ensure Python compatibility.
# %cp /content/drive/MyDrive/BR_mod.csv ./

df = pd.read_csv("BR_mod.csv")

df

df.columns

df.median()

for column in df.columns:
    df[column].fillna(df[column].median(), inplace=True)

df

data = df.to_numpy()

type(data)

length = data.shape[0]
print(length)

k_values = np.arange(2, 31, 1)
print(k_values)
inertia = []

def clustering(K):
  
  centroid = (df.sample(n=K)).to_numpy()    # Randomly select K centroid
  Y = np.zeros(length)
  iter = 0
  while(1):
    Z = np.zeros(length)      # Category of each point now
    for i, x in enumerate(data):
      dist = np.zeros(K)
      for j, y in enumerate(centroid):
        dist[j] = np.linalg.norm(y - x)
      Z[i] = np.argmin(dist)
    
    temp = np.linalg.norm(Y - Z)      #Difference between previous clustering and current clustering
    if (temp == 0):
      break
    Y = Z                     # Copy the new category

    for i in range(K):        # Calculating new centriod
      sum = np.zeros(24)
      count = 0
      for j, x in enumerate(data):
        if (Y[j] == i):
          sum = sum + x
          count += 1
      centroid[i] = sum/count
    
    iter += 1
  
  print(iter)              #Iteration Number
    
  sum = 0
  for i, x in enumerate(data):
    j = int (Y[i])
    y = centroid[j]
    sum += np.linalg.norm(y - x)
  sum = sum/length
  inertia.append(sum)

for k in k_values:
  clustering(k)

import matplotlib.pyplot as plt
plt.plot(k_values, inertia, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

iter = 0
while(1):
  Z = np.zeros(length)      # Category of each point now
  for i, x in enumerate(data):
    dist = np.zeros(K)
    for j, y in enumerate(centroid):
      dist[j] = np.linalg.norm(y - x)
    Z[i] = np.argmin(dist)
  temp = np.linalg.norm(Y - Z)
  
  if (temp == 0):
    break
  Y = Z                     # Copy the new category

  for i in range(K):        # Calculating new centriod
    sum = np.zeros(24)
    count = 0
    for j, x in enumerate(data):
      if (Y[j] == i):
        sum = sum + x
        count += 1
    centroid[i] = sum/count
  
  print(iter)              #Iteration Number
  iter += 1