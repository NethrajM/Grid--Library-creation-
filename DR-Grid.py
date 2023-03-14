#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# In[2]:


dataset =pd.read_csv('insurance_pre.csv')


# In[3]:


dataset


# In[4]:


dataset=pd.get_dummies(dataset,drop_first=True)


# In[5]:


dataset


# In[6]:


indep=dataset[['age','bmi','children','sex_male','smoker_yes']]
dep=dataset['charges']


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(indep,dep,test_size=50,random_state=0)


# In[8]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[9]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
param_grid={'criterion':['mse','mae','friedman_mse'],
            'max_features':['auto','sqrt','log2'],
            'splitter':['best','random']}
grid = GridSearchCV(DecisionTreeRegressor(), param_grid, refit=True, verbose=3, n_jobs=-1)

grid.fit(x_train, y_train)            


# In[12]:


re=grid.cv_results_
grid_predictions=grid.predict(x_test)
from sklearn.metrics import r2_score 
r_score=r2_score(y_test,grid_predictions)

print("the R_score value for best parameter {}:".format(grid.best_params_),r_score)


# In[14]:


table = pd.DataFrame.from_dict(re)
  


# In[15]:


table


# In[16]:


age_input=float(input("Age:"))
bmi_input=float(input("BMI:"))
children_input=float(input("children:"))
sex_male_input=int(input("sex male 0 or 1:"))
smoker_yes_input=int(input("smoker yes 0 or 1:"))


# In[20]:


Future_Prediction=grid.predict([[age_input,bmi_input,children_input,sex_male_input,smoker_yes_input]])
 
print("Future_Prediction={}".format(Future_Prediction))


# In[ ]:




