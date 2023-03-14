#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[15]:


dataset =pd.read_csv('insurance_pre.csv')


# In[16]:


dataset


# In[17]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[18]:


dataset


# In[19]:


indep=dataset[['age','bmi','children','sex_male','smoker_yes']]
dep=dataset['charges']


# In[21]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {'criterion': ['mse', 'mae'],
              'max_features': ['auto', 'sqrt', 'log2'],
              'n_estimators':[10,100]}

grid=GridSearchCV(RandomForestRegressor(), param_grid, refit=True, verbose=3, n_jobs=-1)
grid.fit(indep,dep)

     


# In[24]:


re=grid.cv_results_
print("The R_score value for best parameter{}:".format(grid.best_params_))


# In[25]:


table = pd.DataFrame.from_dict(re)
  


# In[26]:


table


# In[27]:


age_input=float(input("Age:"))
bmi_input=float(input("BMI:"))
children_input=float(input("children:"))
sex_male_input=int(input("sex male 0 or 1:"))
smoker_yes_input=int(input("smoker yes 0 or 1:"))


# In[28]:


Future_Prediction=grid.predict([[age_input,bmi_input,children_input,sex_male_input,smoker_yes_input]])
 
print("Future_Prediction={}".format(Future_Prediction))


# In[ ]:





# In[ ]:




