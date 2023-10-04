#!/usr/bin/env python
# coding: utf-8

# # -- Stock Price Prediction of Netflix using LSTM --

# # Import the necessary liabraries

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# # load the data

# In[34]:


df=pd.read_csv("C:/Mechatronics/Payroll/netflix.csv")


# In[35]:


df


# In[36]:


df.shape


# In[37]:


df.info()


# # drop the columns with missing values

# In[38]:


df.isna().sum()


# # description of data

# In[39]:


df.describe()


# In[40]:


#converting "date" dtype column from object to date
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[41]:


df


# # create a dataframe with only the 'Close prices'

# In[42]:


data=pd.DataFrame(df['Close'])
data


# # create a new column for target variables

# In[43]:


df['Next_Close']=df['Close'].shift(-1)
df


# # split the data into training and testing data

# In[44]:


x=df[['Close']]
x


# In[45]:


y=df[['Next_Close']]
df


# In[46]:


x_train,x_test,y_train,y_test = train_test_split ( x ,y , test_size = 0.3 ,random_state = 42)


# In[ ]:





# # create a Linear Regression model

# In[47]:


model = LinearRegression()


# # fit the model on training data

# In[48]:


model.fit(x_train , y_train)


# # make predictions on test data

# In[49]:


y_pred = model.predict(x_test)


# In[50]:


y_pred


# In[ ]:





# # visualize the actual vs predicted value

# In[58]:


plt.scatter (x_test ,y_test ,color ='blue')
plt.plot (x_test ,y_pred ,color= 'red',linewidth =2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual VS Predicted Stock Price')
plt.show


# In[ ]:




