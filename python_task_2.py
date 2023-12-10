#!/usr/bin/env python
# coding: utf-8

# # **<blockquote style="color:#0047AB; font-family: Arial, sans-serif;">MapUp Data Analyst Assessment</blockquote>**
# 
# ## **<span style="color:hotpink; font-family: Arial, sans-serif;">python_task_2</span>**
# 
# ### **Name:** SUSHIL PRASAD BOOPATHY M

# ### <span style="color:hotpink;">Importing the necessary libraries</span>

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ### <span style="color:hotpink;">Question 1: Distance Matrix Calculation</span>

# In[4]:


def calculate_distance_matrix(df):
    
    toll_locations = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    

    distance_matrix = pd.DataFrame(index=toll_locations, columns=toll_locations)
    
    
    distance_matrix = distance_matrix.fillna(0)

    for index, row in df.iterrows():
    
        toll_A, toll_B, distance = row['id_start'], row['id_end'], row['distance']
        
        distance_matrix.at[toll_A, toll_B] += distance
        
        distance_matrix.at[toll_B, toll_A] += distance

    distance_matrix.values[[range(len(distance_matrix))]*2] = 0
    
    return distance_matrix

df1 = pd.read_csv('dataset-3.csv')
dist_matrix = calculate_distance_matrix(df1)
dist_matrix = dist_matrix.dropna() 

print(dist_matrix)


# ### <span style="color:hotpink;">Question 2: Unroll Distance Matrix</span>

# In[6]:


def unroll_distance_matrix(result_matrix):

    toll_locations = result_matrix.columns

   
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    
    for start_loc in toll_locations:
        for end_loc in toll_locations:
            
            if start_loc != end_loc:
                
                unrolled_df = unrolled_df.append({
                    'id_start': start_loc,
                    'id_end': end_loc,
                    'distance': result_matrix.at[start_loc, end_loc]
                }, ignore_index=True)

    return unrolled_df
condition = lambda x: x ==0


unrolled_mtrx = unroll_distance_matrix(dist_matrix)


print(unrolled_mtrx)


# ### <span style="color:hotpink;">Question 3: Finding IDs within Percentage Threshold</span>

# In[9]:


def find_ids_within_ten_percentage_threshold(df):
   
    overall_avg_distance = df['distance'].mean()

    
    lower_threshold = overall_avg_distance - (overall_avg_distance * 0.1)
    upper_threshold = overall_avg_distance + (overall_avg_distance * 0.1)

    
    result_df = df.groupby('id_start')['distance'].mean().reset_index()
    result_df = result_df[(result_df['distance'] >= lower_threshold) & (result_df['distance'] <= upper_threshold)]

    
    result_ids = sorted(result_df['id_start'].unique())

    return result_ids


result_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_mtrx)

print(result_within_threshold)


# ### <span style="color:hotpink;">Question 4: Calculate Toll Rate</span>

# In[11]:


def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.1,
        'car': 0.2,
        'rv': 0.3,
        'bus': 0.4,
        'truck': 0.5
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}_toll'
        df[column_name] = df['distance'] * rate_coefficient

    return df

result_with_toll_rates = calculate_toll_rate(unrolled_mtrx)

print(result_with_toll_rates)


# ### <span style="color:hotpink;">Question 5: Calculate Time-Based Toll Rates</span>

# In[13]:


import pandas as pd
from datetime import time



def calculate_time_based_toll_rates(df):
    
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]

    
    weekend_discount_factor = 0.7

    
    df['start_time'] = pd.to_datetime(df['id_start']).dt.time
    df['end_time'] = pd.to_datetime(df['id_end']).dt.time


    for index, row in df.iterrows():
        
        start_day = row['id_start']
        end_day = row['id_end']

        
        discount_factor = weekend_discount_factor if start_day in [6, 7] else None 

        for start, end, factor in time_ranges:
            if start <= row['start_time'] <= end and start <= row['end_time'] <= end:
                discount_factor = factor

        
        for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
            column_name = f'{vehicle_type}_toll'
            result_with_toll_rates.at[index, column_name] *= discount_factor

    return df



result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rates)
print(result_with_time_based_toll_rates)

