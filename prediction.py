#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import RandomizedSearchCV , train_test_split


# In[2]:


df = pd.read_excel(r"C:\Users\Admin\Desktop\air\air_train.xlsx",engine='openpyxl')
df1 = pd.read_excel(r"C:\Users\Admin\Desktop\air\air_test.xlsx",engine='openpyxl')

df.head()


# In[3]:


df.dropna(inplace = True)
df1.dropna(inplace = True)


# In[4]:


#Converting string values in columns "Date_of_Journey", "Dep_Time" and"Arrival_Time" to Datetime

#For train dataset

df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"]).dt.month
df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"]).dt.day

df["Dep_Time_hr"] = pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_Time_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute

df["Arrival_Time_hr"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
df["Arrival_Time_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute

#For test dataset

df1["Journey_month"] = pd.to_datetime(df1["Date_of_Journey"]).dt.month
df1["Journey_day"] = pd.to_datetime(df1["Date_of_Journey"]).dt.day

df1["Dep_Time_hr"] = pd.to_datetime(df1["Dep_Time"]).dt.hour
df1["Dep_Time_min"] = pd.to_datetime(df1["Dep_Time"]).dt.minute

df1["Arrival_Time_hr"] = pd.to_datetime(df1["Arrival_Time"]).dt.hour
df1["Arrival_Time_min"] = pd.to_datetime(df1["Arrival_Time"]).dt.minute


# In[5]:


df.drop(["Date_of_Journey" , "Dep_Time" , "Arrival_Time","Additional_Info","Route"] , axis =1 , inplace = True)
df1.drop(["Date_of_Journey" , "Dep_Time" , "Arrival_Time","Additional_Info","Route"] , axis =1 , inplace = True)


# In[6]:


#Converting Categorical Features into numerical form using LabelEncoder()

le = LabelEncoder()

#For train dataset

df["Source"] = le.fit_transform(df["Source"])
df["Destination"] = le.fit_transform(df["Destination"])

#For test dataset

df1["Source"] = le.fit_transform(df1["Source"])
df1["Destination"] = le.fit_transform(df1["Destination"])


# In[7]:


# Determining no. of flights for different Airline companies in train and test dataset respectively

print(df.Airline.value_counts())   # For train dataset
print(" ")
print(df1.Airline.value_counts())  # For test dataset


# In[8]:


#mapping no. of stops for train and test dataset respectively

stop = {
    "non-stop":0,
    "1 stop":1,
    "2 stops":2,
    "3 stops":3,
    "4 stops":4
}

df.loc[: , "Total_Stops"] = df["Total_Stops"].map(stop)    # For train dataset

df1.loc[: , "Total_Stops"] = df1["Total_Stops"].map(stop)  # For test dataset


# In[9]:


#Loading train dataset
df.head()


# In[10]:


#change of duration into hr and min
#train

duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[11]:


#test
duration = list(df1["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours_t = []
duration_mins_t = []
for i in range(len(duration)):
    duration_hours_t.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins_t.append(int(duration[i].split(sep = "m")[0].split()[-1]))


# In[12]:


df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins


# In[13]:


df1["Duration_hours"] = duration_hours_t
df1["Duration_mins"] = duration_mins_t


# In[14]:


df.head(8) 


# In[15]:


df1.head() 


# In[16]:


df.drop(["Duration"] , axis =1 , inplace =True)
df1.drop(["Duration"] , axis =1 , inplace =True)


# In[17]:


stop = {
    "Jet Airways":1,
    "IndiGo":2,
    "Air India":3,
    "Multiple carriers":4,
    "SpiceJet":5 , "Vistara":6 ,"Air Asia":7 , "GoAir":8, 
}

df.loc[: , "Airline"] = df["Airline"].map(stop)
df1.loc[: , "Airline"] = df1["Airline"].map(stop)


# In[18]:


for name in stop.values():
    print(name)


# In[19]:


df.isna().sum()


# In[20]:


df = df[df.Airline != 'Trujet']

df = df[df.Airline != 'Multiple carriers Premium economy']
df = df[df.Airline != 'Jet Airways Business']
df = df[df.Airline != 'Vistara Premium economy']



df1 = df1[df1.Airline != 'Multiple carriers Premium economy']
df1 = df1[df1.Airline != 'Jet Airways Business']
df1 = df1[df1.Airline != 'Vistara Premium economy']


# In[21]:


df.isna().sum()


# In[22]:


df.dropna(inplace = True)
df1.dropna(inplace = True)


# In[23]:


df.isna().sum()


# In[24]:


df1.isna().sum()


# In[25]:


x = df.drop(["Price"] , axis =1)
y = df.Price
x_train , x_test , y_train , y_test = train_test_split(x,y,random_state = 100 , test_size = 0.3)


# In[26]:


feat = ExtraTreesRegressor()
feat.fit(x_train , y_train)


# In[27]:


features = pd.Series( feat.feature_importances_ , index = x_train.columns )
features.nlargest(10).plot(kind = "barh")
plt.show()


# In[28]:


##create model
lr = LinearRegression()
xgb = XGBRegressor()
rfr = RandomForestRegressor()
dt = DecisionTreeRegressor()


# In[29]:


print(lr.fit(x_train , y_train))
print(xgb.fit(x_train , y_train))
print(rfr.fit(x_train , y_train))
print(dt.fit(x_train , y_train))


# In[30]:


print(r2_score(lr.predict(x_train) , y_train))
print(r2_score(xgb.predict(x_train) , y_train))
print(r2_score(rfr.predict(x_train) , y_train))
print(r2_score(dt.predict(x_train) , y_train))


# In[31]:


sb.distplot(rfr.predict(x_train) - y_train)


# In[32]:


sb.distplot(rfr.predict(x_test) - y_test)


# In[33]:


rf_p = {
    
    "min_samples_split": list(range(2,11)),
    "min_samples_leaf" : list(range(1,10)),
    "max_depth":list(range(1,200)),
    "n_estimators": list(range(1,500))
}

dt_p = {
    "criterion":["mse"],
    "min_samples_split": list(range(2,11)),
    "min_samples_leaf" : list(range(1,10)),
    "max_depth":list(range(1,200))
}

xgb_p = {
    "learning_rate" : [0.1,0.2,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.5],
    "max_depth" : list(range(1,200)),
    "booster" : ['gbtree', 'gblinear' ,'dart'],
    "min_child_weight" : list(range(1,20)),
    "n_estimators" : list(range(1,200))
}


# In[34]:


rscv = RandomizedSearchCV(rfr , param_distributions=rf_p , cv =10 , n_iter=10  ,n_jobs = -1 , verbose = 10)


# In[35]:


rscv.fit(x,y)


# In[36]:


rscv.best_estimator_


# In[37]:


rfr =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=35,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=3, min_samples_split=9,
                      min_weight_fraction_leaf=0.0, n_estimators=69,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)


# In[38]:


rfr.fit(x_train , y_train)
xgb.fit(x_train , y_train)
dt.fit(x_train , y_train)


# In[39]:


print(r2_score(rfr.predict(x_test) , y_test))
print(r2_score(xgb.predict(x_test) , y_test))
print(r2_score(dt.predict(x_test) , y_test))


# In[40]:


df1


# In[41]:


file = open(r'C:\Users\Admin\Desktop\air\main_flight_rfr.pkl', "wb")
pickle.dump(rfr , file)


# In[42]:


model = open(r'C:\Users\Admin\Desktop\air\main_flight_rfr.pkl', "rb")
forest = pickle.load(model)


# In[43]:


z = forest.predict(df1.iloc[1:2 , :])


# In[44]:


for j in z:
    print(j)


# In[ ]:




