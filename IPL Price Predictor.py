#!/usr/bin/env python
# coding: utf-8

# ### **Connect With Me in Linkedin** :- https://www.linkedin.com/in/dheerajkumar1997/

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# # Load Data

# In[35]:


df = pd.read_csv("D:\Data Science Python Programs\IPL-First-Innings-Score-Prediction-Deployment-master\ipl.csv")
df


# # Exploratoray Data Aalysis

# In[19]:


def Show_DisPlot(x):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,7))
    sns.distplot(x, bins = 25)


# In[22]:


Show_DisPlot(df.runs)


# In[23]:


Show_DisPlot(df.wickets)


# In[24]:


Show_DisPlot(df.overs)


# In[25]:


Show_DisPlot(df.runs_last_5)


# In[27]:


Show_DisPlot(df.total)


# In[28]:


def Show_CountPlot(x):
    fig_dims = (18, 8)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.countplot(x,ax=ax)


# In[30]:


Show_CountPlot(df.bat_team)


# In[31]:


Show_CountPlot(df.bowl_team)


# In[36]:


Show_CountPlot(df.bowler)


# In[37]:


Show_CountPlot(df.batsman)


# # Data Preprocessing

# In[18]:


def show_hist(x):
    plt.rcParams["figure.figsize"] = 15,18
    x.hist()
show_hist(df)


# In[3]:


columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)


# In[4]:


df


# In[5]:


consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
df


# In[6]:


df = df[df['overs']>=5.0]
df


# In[7]:


from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[8]:


encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[9]:


encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[10]:


encoded_df


# In[11]:


def show_hist(x):
    plt.rcParams["figure.figsize"] = 15,18
    x.hist()
show_hist(encoded_df)


# In[13]:


encoded_df['wickets_last_5'] = np.log1p(encoded_df['wickets_last_5'])
encoded_df['bat_team_Chennai Super Kings'] = np.log1p(encoded_df['bat_team_Chennai Super Kings'])
encoded_df['bat_team_Delhi Daredevils'] = np.log1p(encoded_df['bat_team_Delhi Daredevils'])
encoded_df['bat_team_Kings XI Punjab'] = np.log1p(encoded_df['bat_team_Kings XI Punjab'])
encoded_df['bat_team_Mumbai Indians'] = np.log1p(encoded_df['bat_team_Mumbai Indians'])
encoded_df['bat_team_Rajasthan Royals'] = np.log1p(encoded_df['bat_team_Rajasthan Royals'])
encoded_df['bowl_team_Chennai Super Kings'] = np.log1p(encoded_df['bowl_team_Chennai Super Kings'])
encoded_df['bowl_team_Delhi Daredevils'] = np.log1p(encoded_df['bowl_team_Delhi Daredevils'])
encoded_df['bowl_team_Kolkata Knight Riders'] = np.log1p(encoded_df['bowl_team_Kolkata Knight Riders'])
encoded_df['bowl_team_Mumbai Indians'] = np.log1p(encoded_df['bowl_team_Mumbai Indians'])
encoded_df['bowl_team_Kings XI Punjab'] = np.log1p(encoded_df['bowl_team_Kings XI Punjab'])


# # Train Test Split

# In[14]:


X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[15]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[16]:


X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# # Model Building

# In[17]:


import statsmodels.api as sm
model2 =sm.OLS(y_train,X_train).fit()
model2.summary()


# In[38]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
lr_pred = regressor.predict(X_test)


# In[39]:


from sklearn import metrics
LR_RMSE = np.sqrt(metrics.mean_squared_error(y_test,lr_pred))
LR_RMSE 


# In[40]:


from sklearn.metrics import r2_score
LR_r2_score = r2_score(y_test,lr_pred)
LR_r2_score


# In[ ]:


Creating a pickle file for the classifier
filename = 'lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[41]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,random_state=42)
reg=regressor.fit(X_train,y_train)
RF_pred=regressor.predict(X_test)


# In[42]:


from sklearn import metrics
RF_RMSE = np.sqrt(metrics.mean_squared_error(y_test,RF_pred))
RF_RMSE


# In[43]:


from sklearn.metrics import r2_score
RF_r2_score = r2_score(y_test,RF_pred)
RF_r2_score


# ### **Connect With Me in Linkedin** :- https://www.linkedin.com/in/dheerajkumar1997/
