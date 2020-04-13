#%%
import numpy as np
import pandas as pd 
import os
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

stocks=pd.read_csv('./datasets/question2.csv')
stocks=stocks.dropna()
#%%
stocks_date=stocks.date.str.rsplit("-",n=1,expand=True)
stocks_date.columns=['month','year']
stocks=pd.concat([stocks,stocks_date],axis=1)
stocks.month=stocks.month.map({'Jan':1,'Feb':2,'Mar':3,'Apr':4,
'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})

def sort_group(X):
    X=X.sort_values('return',ascending=False)
    X=X.reset_index(drop=True)
    X['assign']=0
    X['assign'][:20]=1
    return X

stocks=stocks.groupby('date').apply(sort_group)
stocks=stocks.reset_index(drop=True)

stocks['year']=pd.to_numeric(stocks.year)
stocks.year=2000+stocks.year   
stocks.date=stocks.year*10000+stocks.month*100+1

# stocks['date_used']=stocks.year*10000+stocks.month_used*100+1

#%%
change=stocks[['assign','month_used','year']]
change.loc[change.month_used==13]['year']=1+change.loc[change.month_used==13]['year']
change.loc[change.month_used==13]['month_used']=1

# %%
change.columns=['assign','month','year']
q2=stocks.drop(['assign','month_used'],axis=1)
q2=pd.merge(q2,change,on=['year','month'],how='left')

# match=q2[['assign','date_used']]

# match.date_used=pd.to_datetime(match.date_used,format='%Y%m%d')
# q2.date=pd.to_datetime(q2.date,format='%Y%m%d')
#%%


# %%
