#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px

# Handling Warnings
warnings.filterwarnings('ignore')

# Importing Data 

transaction_data = pd.read_csv("transactions.csv")
transaction_data['date'] = pd.to_datetime(transaction_data['date'], format = "%Y-%m-%d")

oil_data = pd.read_csv("oil.csv")
oil_data['date'] = pd.to_datetime(oil_data['date'], format = "%Y-%m-%d")

store_data = pd.read_csv("stores.csv")

holiday_data = pd.read_csv("holidays_events.csv")
holiday_data['date'] = pd.to_datetime(holiday_data['date'], format = "%Y-%m-%d")

data = pd.read_csv("train.csv")
data['date'] = pd.to_datetime(data['date'], format = "%Y-%m-%d")


# In[2]:


import pandas as pd

# Set your minimum and maximum dates
min_date = oil_data.date.min()
max_date = oil_data.date.max()

# Generate a range of dates between min and max dates
date_range = pd.date_range(start=min_date, end=max_date, freq='D')

# Create a DataFrame with the dates
full_oil_data = pd.DataFrame({'date': date_range})



full_oil_data = pd.merge(full_oil_data, oil_data, on='date', how='left')

# intepolate the dcoilwitco oilprices column
full_oil_data['dcoilwtico'] = full_oil_data['dcoilwtico'].interpolate(method='nearest')

full_oil_data.loc[0,'dcoilwtico'] = 93.14

full_oil_data.head()


# In[3]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=full_oil_data, x='date', y='dcoilwtico', color='green')
plt.title("Daily Oil Prices")
plt.xlabel("Date")
plt.ylabel("Oil Price (dcoilwtico)")
plt.tight_layout()
plt.show()


# In[4]:


# data.store_nbr.nunique() # 54
# data.date.min(), data.date.max() # Timestamp('2013-01-01 00:00:00'), Timestamp('2017-08-15 00:00:00')


# In[5]:


#from sklearn.linear_model import LinearRegression


# In[6]:


# model = LinearRegression()
# model.fit(X_train, y_train)
# y_test_preds = model.predict(X_test)


# In[7]:


transaction_data.head()


# In[8]:


oil_data.head()


# In[9]:


store_data.head()


# In[10]:


s1 = set(store_data.city.unique())


# In[11]:


s2 = set(holiday_data.locale_name.unique())


# In[12]:


s1 - s2, s2 - s1


# In[13]:


national_holidays = holiday_data.loc[holiday_data.locale_name=="Ecuador"]
national_holidays = pd.DataFrame({'date':national_holidays.date.unique(), 'is_national_holiday' : 1})
national_holidays.head()


# In[14]:


local_holidays = holiday_data.loc[holiday_data.locale_name!="Ecuador"]
local_holidays = local_holidays[['date', 'locale_name']].drop_duplicates()
local_holidays['is_local_holiday'] = 1


# In[15]:


local_holidays.head()


# In[16]:


holiday_data.type.unique()


# In[17]:


print(data.shape)
data = pd.merge(data, transaction_data, how='left', on =['date', 'store_nbr'])
print(data.shape)
data = pd.merge(data, full_oil_data, how='left', on =['date'])
print(data.shape)
data = pd.merge(data, store_data, how='left', on =['store_nbr'])
print(data.shape)


# In[18]:


print(data.shape)
data = pd.merge(data, national_holidays, how='left', on=['date'])
print(data.shape)


# In[19]:


data.city.value_counts()


# In[20]:


# {'Babahoyo', 'Daule', 'Playas'} in data but not in holidays


# In[21]:


data = pd.merge(data, local_holidays, how='left', left_on=['date', 'city'], right_on=['date', 'locale_name'])
print(data.shape)


# In[22]:


data.is_national_holiday = data.is_national_holiday.fillna(0)
data.is_local_holiday = data.is_local_holiday.fillna(0)

data.locale_name = data.locale_name.fillna("NA")


# In[23]:


data.head()


# In[24]:


_ = plt.boxplot(data.sales, showfliers=False)


# In[25]:


# data.date.min(), data.date.max() # Timestamp('2013-01-01 00:00:00'), Timestamp('2017-08-15 00:00:00')


# In[26]:


split_date = "2017-01-01"


# In[27]:


data.transactions = data.transactions.fillna(0)


# In[28]:


data['is_oil_high'] = data['dcoilwtico'].apply(lambda x: 1 if x>70 else 0)


# In[29]:


train = data.loc[data.date < split_date]
test = data.loc[data.date >= split_date]
print(train.shape, test.shape)


# In[30]:


train.shape[0]  + test.shape[0] == data.shape[0]


# In[31]:


train.head()


# In[32]:


# features

cat_columns = ["store_nbr", 'family', 'city', 'state', 'type', 'cluster']
numerical_cols = ["onpromotion", 'is_national_holiday', 'is_local_holiday', 'is_oil_high']
target = "sales"


# In[33]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# In[34]:


d = {}
for col in cat_columns:
    d[col] = LabelEncoder()
    train[col] = d[col].fit_transform(train[col])
    test[col] = d[col].transform(test[col])


# In[35]:


model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
model.fit(train[cat_columns + numerical_cols], train[target])


# In[36]:


test_preds = model.predict(test[cat_columns + numerical_cols])


# In[37]:


# error we will be using is RMSE

# calculate RMSE between test_preds and test[target] and report the RMSE

from sklearn.metrics import mean_squared_log_error

score = mean_squared_log_error(y_true=test[target] , y_pred=test_preds)


# 1. rmsle without oil price raw feature - 0.5253744189217323
# 1. rmsle with oil price as raw feature - 0.650
# 1. rmsle with feature engineered oil price feature ( > 70, <=70) - 0.46217462375578755
# 

# In[38]:


features = cat_columns + numerical_cols
importances = model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# ref - https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python


# – Find significant variables and intuitively explain their effects (Analysis 
# result)
# 
# Significant variables according to random forest feature importances are family and onpromotion.
# 
# For the family feature - 30% of the total sales is from Grocery, 20% of the sales is from beverages and 11% of the sales is from produce.
# 
# 
# For the onpromotion feature - 
# Interestingly total number of items in a product family that are being promoted has negative impact on total sales.
# 
# 
# 

# In[39]:


family_effect = data.groupby(['family']).agg({'sales': 'sum'})
sum_ = family_effect['sales'].sum()
family_effect.sales = round(family_effect.sales * 100 /sum_, 5)
family_effect = family_effect.sort_values(by='sales', ascending=False)


# In[113]:


family_effect


# In[41]:


### Merging tables 

train1 = data.merge(holiday_data, on = 'date', how='left')
train1 = train1.merge(oil_data, on = 'date', how='left')
train1 = train1.merge(transaction_data, on = ['date','store_nbr'], how = 'left')
train1 = train1.merge(store_data, on = 'store_nbr', how = 'left')
train1 = train1.rename(columns = {"type_x" : "holiday_type", "type_y" : "store_type"})
print(train1.columns)
print(train1)


# In[52]:


train1['date'] = pd.to_datetime(train1['date'])
train1['year'] = train1['date'].dt.year
train1['month'] = train1['date'].dt.month
train1['week'] = train1['date'].dt.isocalendar().week
train1['quarter'] = train1['date'].dt.quarter
train1['day_of_week'] = train1['date'].dt.day_name()
train1[:10]


# In[54]:


onpromotion_effect = data.groupby(['onpromotion']).agg({'sales': 'sum'})
sum_ = onpromotion_effect['sales'].sum()
onpromotion_effect.sales = round(onpromotion_effect.sales * 100 /sum_, 5)
onpromotion_effect = onpromotion_effect.sort_values(by='sales', ascending=False)


# In[55]:


# onpromotion feature analysis
_ = onpromotion_effect.plot()


# In[56]:


# box plot of oil price
_ = plt.boxplot(data.dcoilwtico)


# – Interesting discoveries (Selling points)
# 
# Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices. But directly inclduing oil price as feature into the Random forest model did not improve RMSLE. 
# Conducting feature engineering on the oil price column and establishing a binary feature by applying a threshold of 70, then incorporating this feature into the model, has resulted in an enhancement of the RMSLE. This highlights the significance of feature engineering.
# 
# 
# 

# – What you have learned from the real data analysis: data cleansing, missing data, data transformations, like normalization, taking log and adding square terms (Discussion)
# 
# - data cleansing and handling missing data are the most time taking part.
# - Data transformations like normalization did not influence the performance of the Random Forest model. (you should know why)
# - Using RMSLE reduces the effect of outliers compared to RMSE.
# 

# In[64]:


plot1 = px.line(transaction_data.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr',title = "Transactions" )
plot1.show()


# In[57]:


a = transaction_data.copy()
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month
plt.figure(figsize=(10, 6))
sns.boxplot(data=a, x="year", y="transactions", hue="month")
plt.title("Transactions")
plt.xlabel('Year')
plt.ylabel('Transactions')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[58]:


a = transaction_data.copy()
a["year"] = a.date.dt.year
a["dayofweek"] = a.date.dt.dayofweek+1
a = a.groupby(["year", "dayofweek"]).transactions.mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=a, x="dayofweek", y="transactions", hue="year", palette="husl")
plt.title("Transactions by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Transactions")
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[59]:


trend = pd.merge(data.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transaction_data, how = "left")


# In[60]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=trend, x="transactions", y="sales")
sns.regplot(data=trend, x="transactions", y="sales", scatter=False, color='red')
plt.title("Scatter Plot with Trendline")
plt.xlabel("Transactions")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()


# In[61]:


trend2 = pd.merge(data.groupby(["date", "store_nbr"]).sales.sum().reset_index(), full_oil_data, how = "left")


# In[63]:


fig, ax = plt.subplots(figsize=(12, 6))

sns.lineplot(data=trend2, x="dcoilwtico", y="sales", ax=ax)
sns.regplot(data=trend2, x="dcoilwtico", y="sales", scatter=False, color='red', ax=ax)
ax.set_title("Average Sales vs Oil Prices")

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




