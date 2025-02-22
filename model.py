#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#dataUpload
df=pd.read_csv(r"Housing.csv",sep=',')
df.head()


# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.isna()


# In[5]:


df.isnull().sum()


# In[6]:


# Encode categorical variables (yes/no to 1/0)
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})
df.head()


# In[7]:


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Encode 'furnishingstatus' using Label Encoding
le = LabelEncoder()
df["furnishingstatus"] = le.fit_transform(df["furnishingstatus"])
df.head()


# In[8]:


# Save the cleaned dataset
df.to_csv("cleaned_housing.csv", index=False)


# In[9]:


# Display cleaned dataset sample
df.head()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

#Correlation Heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split

# Split data
X = df.drop(columns=['price', 'guestroom','basement', 'hotwaterheating', 'furnishingstatus'])
y = df['price']/100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


# In[14]:


y_train


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate model
y_pred = model.predict(X_test)
mae=mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')


# In[16]:


import joblib
# Save model
joblib.dump(model, 'model.pkl')


# In[ ]:




