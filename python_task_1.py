#!/usr/bin/env python
# coding: utf-8

# # **<blockquote style="color:#0047AB; font-family: Arial, sans-serif;">MapUp Data Analyst Assessment</blockquote>**
# 
# ## **<span style="color:hotpink; font-family: Arial, sans-serif;">python_task_1</span>**
# 
# ### **Name:** SUSHIL PRASAD BOOPATHY M

# ### <span style="color:hotpink;">Importing the necessary libraries</span>

# In[232]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ### <span style="color:hotpink;">Question 1: Car Matrix Generation</span>

# In[233]:


def generate_car_matrix(df):
    
    df = pd.read_csv(df)

    
    mtrx_df = df.pivot(index='id_1', columns='id_2', values='car')

    
    mtrx_df = mtrx_df.fillna(0)

    
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            if i==j:
                df.loc[i,j] = 0
    
    
    return mtrx_df
df = 'dataset-1.csv'
car_matrix = generate_car_matrix(df)
print(car_matrix)


# ### <span style="color:hotpink;">Question 2: Car Type Count Calculation</span>

# In[234]:


import pandas as pd

def get_type_count(df):

    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)
    
    
    counts = df['car_type'].value_counts().to_dict()

    
    asc_keys = dict(sorted(counts.items()))

    
    return asc_keys



df = pd.read_csv('dataset-1.csv')
type_count = get_type_count(df)
print(type_count)


# ### <span style="color:hotpink;">Question 3:Bus Count Index Retrieval</span>

# In[235]:


def get_bus_indexes(df):
    
    bus_indexes = df[df['bus'] > 2 * df['bus'].mean()].index.tolist()

    
    bus_indexes.sort(reverse = False)

    
    return bus_indexes

df = pd.read_csv('dataset-1.csv')
bus_count = get_bus_indexes(df)
print(bus_count)


# ### <span style="color:hotpink;">Question 4: Route Filtering</span>
# 

# In[236]:


def filter_routes(df):
    
    truck_avg = df.groupby('route')['truck'].mean().sort_values(ascending= False)

    
    routes = truck_avg[truck_avg > 7].index.tolist()


    return routes

df = pd.read_csv('dataset-1.csv')
result = filter_routes(df)
print(result)


# ### <span style="color:hotpink;">Question 5: Matrix Value Modification</span>

# In[237]:


def multiply_matrix(df):
    
    
    mult_df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    
    modified_df = mult_df.round(1)

    
    return modified_df



df = pd.DataFrame(car_matrix)
mult_mtrx = multiply_matrix(df)
print(mult_mtrx)

