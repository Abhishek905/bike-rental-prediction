#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[2]:


os.chdir('E:/documents')
os.getcwd()


# In[3]:


df=pd.read_csv('bikes.csv')


# In[4]:


copy=df.copy()


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


# check the datatypes
df.dtypes


# In[8]:


# check the no. of unique values in each variable 
for i in df.columns:
    print('no of unique values in {} : {}'.format(i,df[i].nunique()))


# we have to convert the datatypes of season, yr, mnth, holiday, weekday, workingday and weathersit 

# In[9]:


# first change the data types of variables into proper type
col=['season','yr','mnth','holiday','weekday','workingday','weathersit']
for i in col:
    df[i]=df[i].astype('object')


# In[10]:


df['dteday']=pd.to_datetime(df['dteday'])


# In[11]:


def get_date(dt):
    return dt.day


# In[12]:


df['date']=df['dteday'].map(get_date)


# In[13]:


df.dtypes
# we can see that the datatypes are changed to proper ones


# In[14]:


# check for duplicate elements to avoid data leakege
df.duplicated().sum()


# we can see that there is no duplicate elements in the dataset

# In[15]:


# missing value analysis
df.isna().sum()


# In[16]:


# visualize the missing values in the dataset
import missingno
missingno.matrix(df)


# There is no missing values in the dataset

# In[17]:


df.head()


# we know that the causal,registered and cnt are the dependent variables and also cnt is the sum of casual and registered variables so we can drop the casual and registered variables

# In[18]:


df=df.drop(['casual','registered'],axis=1)


# In[19]:


df.shape


# # Exploratory data analysis

# In[20]:


df.dtypes


# In[21]:


df.shape


# In[22]:


# seperate the numerical and categorical variables
df_num=df.select_dtypes(exclude=['object','datetime64'])
df_num_col=df_num.columns


# In[23]:


df_num.shape


# In[24]:


df_cat=df.select_dtypes(include='object')
df_cat_col=df_cat.columns


# In[25]:


df_cat.shape


# In[26]:


df_num_col


# In[27]:


df_col=['temp', 'atemp', 'hum', 'windspeed', 'cnt']


# # Univariate analysis

# In[28]:


sns.distplot(df['temp'],kde=True,color='red')


# In[29]:


sns.distplot(df['atemp'],kde=True,color='black')


# we can observe that the temp and atemp distributions are almost same 

# In[30]:


sns.distplot(df['hum'],kde=True)


# It resembles the normal distribution with almost neglegeble outliers

# In[31]:


sns.distplot(df['windspeed'],kde=True,color='green')


# windspeed also resembles the normal distribution with some outliers 

# In[32]:


sns.distplot(df['cnt'],kde=True)


# Dependent variable should follow normal distribution and it almost resembles the normal distribution

# # Bivariate analysis

# In[33]:


df.columns


# Dealing with categorical variables

# In[34]:


graph=sns.barplot(x='season',y='cnt',data=df)


# we can observe the interesting pattern. There is a high demand in fall and low demand in spring

# In[35]:


sns.barplot(x='yr',y='cnt',data=df)


# Year does have an effect on demand but we have only two years data. we don't know what will happen in the next year so better to drop this variable in order to avoid confusion to the algorithm

# In[36]:


sns.barplot(x='mnth',y='cnt',data=df)


# Month does have an effect on demand. It is usually high in the months of summer

# In[37]:


sns.barplot(x='holiday',y='cnt',data=df)


# we can observe the interesting pattern here. Demand is higher in the working days because most of the working people are using the bikes to reach the office

# In[38]:


sns.barplot(x='weekday',y='cnt',data=df)


# Weekday does not have any effect on demand. Its almost same in all the days. so it's better to drop this variable

# In[39]:


sns.barplot(x='workingday',y='cnt',data=df)


# we have observed the same type of result in holiday variable. It might create multicollinearity problem. so it's better to drop one of the variable to avoid redundant variables in the dataset. 

# In[40]:


sns.barplot(x='date',y='cnt',data=df)


# date doesn't have any effect on demand variable. so it's better to drop that date variable also.

# In[41]:


sns.barplot(x='weathersit',y='cnt',data=df)


# Demand is much dependent on weathersit. clearly the demand is low when the climate is cool with rain and thunderstorm and higher when the weather is clear.

# In[42]:


df_num.columns


# Dealing with numerical variables

# In[43]:


sns.relplot(x='instant',y='cnt',data=df)


# In[44]:


sns.relplot(x='temp',y='cnt',data=df)


# Temp has an effect on demand. As temp increases demand also increases.

# In[45]:


sns.relplot(x='atemp',y='cnt',data=df)


# we can observe that the atemp pattern is same as temp pattern. It's better to check the correlation between these variable and decide whether to remove that variable or not. 

# In[46]:


sns.relplot(x='hum',y='cnt',data=df)


# We can't come to the conclusion by only seeing this graph. so we have check for correlation  

# In[47]:


sns.relplot(x='windspeed',y='cnt',data=df)


# we can observe that the demand is decreasing when the windspeed increases

# In[48]:


# drop the variables which are not useful in predicting the target variable.
col_drop=['yr','date','instant','dteday']
df=df.drop(col_drop,axis=1)


# In[49]:


df.head()


# # Check for correlation 

# In[50]:


corr=df[['temp','atemp','hum','windspeed','cnt']].corr()


# In[51]:


corr


# In[52]:


sns.heatmap(corr)


# we should remove atemp variable because it is highly correlated with the temp variable. If we include this variable in the dataset we should face a multicollinearity problem.

# In[53]:


df=df.drop('atemp',axis=1)


# In[54]:


df.head()


# # chisquare test for categorical features

# In[55]:


col = ['season','mnth','holiday','workingday']


# In[56]:


pip install scipy


# In[57]:


from scipy.stats import chi2_contingency


# In[58]:


for i in col:
    for j in col:
        if(i==j):
            continue
        else:
            chi2,p,dof,ex=chi2_contingency(pd.crosstab(df[i],df[j]))
            print('p value between {} and {} is {}'.format(i,j,p))


# Null hypothesis(ho):Two variables are independent  
# Alternate hypothesis(h1): Two variables are not independent   
# If p<0.05 we will reject the Null hypothesis saying that two variables are dependent on each other.
# We can see that the dependency between holiday and the workingday is high. so it's better to drop one variable

# # Deal with outliers

# In[59]:


sns.catplot(x='temp',data=df,kind='box',orient='v')


# There are no outliers in the temp variable

# In[60]:


sns.catplot(x='windspeed',data=df,orient='v',kind='box')


# In[61]:


#Function to remove outliers
def outliers(variable):
    low=0.01
    high=0.99
    new=df.quantile([low,high])
    l=new[variable][low]
    print('low-{}'.format(l))
    h=new[variable][high]
    print('high-{}'.format(h))
    lc=len(df[df[variable]<=l])
    hc=len(df[df[variable]>=h])
    t=len(df)
    print('percentage of outliers-{}'.format(((lc+hc)/t)*100))
    #assigning the higher and the lower values in the place of outliers
    df.loc[df[variable]<l,variable]=l
    df.loc[df[variable]>h,variable]=h


# In[62]:


outliers('windspeed')


# In[63]:


sns.catplot(x='windspeed',data=df,orient='v',kind='box')


# we can see that the outliers are removed to some extent

# In[64]:


sns.catplot(x='hum',data=df,orient='v',kind='box')


# In[65]:


df.loc[df['hum']<0.22,'hum']=0.22


# In[66]:


sns.catplot(x='hum',data=df,kind='box',orient='v')


# we can see that the outliers are removed 

# # normality check for the target variable

# In[67]:


# check for the normality of target variable
plt.hist(df['cnt'])
("")


# In[68]:


sns.distplot(df['cnt'])


# Target variable resembles the normal distribution so we don't have to change the target data

# # Autocorrelation in the target variable

# In[69]:


# check for autocorrelation in target variable because it is a time series data
# first convert the cnt variable into float 
df1=pd.to_numeric(df['cnt'],downcast='float')


# In[70]:


# plot the autocorrelation plot
plt.acorr(df1,maxlags=10)
("")


# we can see that there is a high autocorrelation in the target variable 

# In[71]:


time_1=df['cnt'].shift(+1).to_frame()


# In[72]:


time_2=df['cnt'].shift(+2).to_frame()


# In[73]:


time_3=df['cnt'].shift(+3).to_frame()


# In[74]:


time_4=df['cnt'].shift(+4).to_frame()


# In[75]:


time_con=pd.concat([time_1,time_2,time_3,df['cnt']],axis=1)


# In[76]:


time_con.columns=['time_1','time_2','time_3','cnt']


# In[77]:


time_con.corr()


# We can see that there is a high correlation with the past values. So we can use these values as the independent variabels to predict the target variable.

# In[78]:


df.drop('cnt',axis=1,inplace=True)


# In[79]:


final_df=pd.concat([df,time_con],axis=1)


# In[80]:


final_df.head()


# In[81]:


final_df.dropna(inplace=True)


# # one hot encoding 

# In[82]:


final_df.head()


# In[83]:


final_df.dtypes
# dummies will create only when the variable is object type


# In[84]:


final_df=pd.get_dummies(final_df,drop_first=True)


# In[85]:


final_df.head()


# In[86]:


final_df.shape


# # Standardize the data

# In[87]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
norm=scaler.fit_transform(final_df)


# In[88]:


norm=pd.DataFrame(norm)


# In[89]:


norm.columns=final_df.columns


# In[90]:


norm.head()


# In[91]:


# split the independent and dependent variables
x=norm.drop('cnt',axis=1)
y=norm['cnt']


# In[92]:


# spliting the data into train and test
# since it is a time series data we have to split differently 
train_size=0.7*len(final_df)
train_size=int(train_size)

x_train=x.iloc[0:train_size,:]
x_test=x.iloc[train_size:,:]

y_train=y.iloc[0:train_size,]
y_test=y.iloc[train_size:,]


# In[93]:


x_train.head()


# In[94]:


x_test.head()


# # appling the different algorithms

# In[95]:


import statsmodels.api as sm


# In[96]:


model=sm.OLS(y_train,x_train).fit()


# In[97]:


model.summary()


# In[98]:


pre=model.predict(x_test)


# In[99]:


from sklearn.metrics import mean_squared_error


# In[100]:


np.sqrt(mean_squared_error(y_test,pre))


# In[102]:


from sklearn.metrics import mean_absolute_error


# In[103]:


mean_absolute_error(y_test,pre)


# In[ ]:





# In[ ]:




