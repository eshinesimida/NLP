# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:17:33 2018

@author: admin
"""

import pandas as pd

#read  csv file
df = pd.read_csv('ZILLOW-M550_SALES.csv')

def arrange_time_dataframe(df):
    df.columns = ['date','value']
    df.sort_values(by = 'date', inplace = True)
    df.set_index('date', inplace = True)
    return df

df = arrange_time_dataframe(df)

ax = df.plot(figsize = (16, 6))
fig = ax.get_figure()
fig.savefig('fig1.png')

#read json file
import json
with open('M550_SALES.json') as f:
    data = json.load(f)
    
print(json.dumps(data, indent = 2))

df1 = pd.DataFrame(data['dataset']['data'])

df1.head()
df1 = arrange_time_dataframe(df1)
df1.plot(figsize = (16, 6))
