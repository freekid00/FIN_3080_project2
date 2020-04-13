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

# %%
stocks=stocks.dropna()

# %%
stocks_date=stocks.date.str.rsplit("-",n=1,expand=True)
stocks_date.columns=['month','year']

# %%
stocks=pd.concat([stocks,stocks_date],axis=1)

# %%
group_decision=stocks.loc[stocks.year=='06']
group_decision=group_decision.groupby('permno').agg({'market_value':np.mean}).reset_index()
group_decision=group_decision.sort_values('market_value',ascending=False).reset_index(drop=True)
group_decision=group_decision.iloc[47:1247,:]

# %%
def group_cut(X):
    X['group']=1
    for i in range(1,10):
        X['group'][i*120:(i+1)*120]=i+1
    return X

group=group_cut(group_decision)
group=group.reset_index(drop=True)
group=group[['permno','group']]
# %%
q1=pd.merge(group,stocks,on='permno',how='left')

# %%
def cal_mon_return(X):
    X['x_sum']=sum(X['market_value'])
    X['ratio']=X['market_value']/X['x_sum']
    X['return']=X['return']*X['ratio']
    return sum(X['return'])
q1=q1.groupby(['group','date']).apply(cal_mon_return)
#%%
q1=pd.DataFrame(q1)
q1=q1.reset_index()
q1.columns=['group','date','return']
# %%
group_date=q1.date.str.rsplit("-",n=1,expand=True)
group_date.columns=['month','year']

# %%
q1=pd.concat([q1,group_date],axis=1)

# %%
q1.month=q1.month.map({'Jan':1,'Feb':2,'Mar':3,'Apr':4,
'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12})

# %%
def cumprod_cal(X):
    X['return']=X['return']+1
    X=X.sort_values('month')
    X['total_return']=X['return'].cumprod()
    return X

q1=q1.groupby(['group','year']).apply(cumprod_cal)
q1=q1.reset_index(drop=True)
q1.year=pd.to_numeric(q1.year)
q1.year=2000+q1.year   
q1.date=q1.year*10000+q1.month*100+1
# %%
q1.date=pd.to_datetime(q1.date,format='%Y%m%d')

#%%
ceshi=pd.pivot_table(q1,values='total_return',columns=['group'],index=['date'])
ceshi.plot(linewidth=0.8)
plt.title('ten portfolios')
plt.ylabel('total return')
plt.legend()
# %%
g1=q1.query('group==1')
g2=q1.query('group==2')
g9=q1.query('group==9')
g10=q1.query('group==10')
# %%
plt.plot(g1['date'],g1['total_return'],'-',color='black',label='value_1',linewidth=1)
plt.plot(g2['date'],g2['total_return'],'-',color='red',label='value_2',linewidth=1)
plt.plot(g9['date'],g9['total_return'],'-',color='green',label='value_9',linewidth=1)
plt.plot(g10['date'],g10['total_return'],'-',color='blue',label='value_10',linewidth=1)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Cumulative Return')
plt.title('Fir two and Last two market_value companies')
# %%
q1.to_csv('./datasets/question2_part1.csv')

# %%
stocks=pd.read_csv('./datasets/question2.csv')
stocks=stocks.dropna()
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

stocks['month_used']=stocks['month']+1
# %%
def cal_mon_return(X):
    X['x_sum']=sum(X['market_value'])
    X['ratio']=X['market_value']/X['x_sum']
    X['return']=X['return']*X['ratio']
    return sum(X['return'])

stocks['year']=pd.to_numeric(stocks.year)
stocks.year=2000+stocks.year   
stocks.date=stocks.year*10000+stocks.month*100+1

# %%
stocks['date_used']=stocks.year*10000+stocks.month_used*100+1
