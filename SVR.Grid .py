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


dataset.columns 


# In[7]:


indep=dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dep=dataset['charges']


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(indep,dep,test_size=50,random_state=0)


# In[9]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler  ()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[10]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
param_grid={'kernel':['rbf','poly','sigmoid','linear'],
            'C':[10,100,1000,2000.3000],'gamma':['auto','scale']}
grid=GridSearchCV(SVR(),param_grid,refit=True,verbose=3,n_jobs=-1)
grid.fit(x_train,y_train)
            


# In[16]:


re=grid.cv_results_
print("The R_score value for best parameter{}:".format(grid.best_params_))
print(grid.best_score_)


# In[17]:


table=pd.DataFrame.from_dict(re)


# In[18]:


table


# In[19]:


age_input=float(input("Age:"))
bmi_input=float(input("BMI:"))
children_input=float(input("children:"))
sex_male_input=int(input("sex Male 0 or1:"))
smoker_yes_input=int(input("smoker yes 0 or 1:"))


# In[27]:


Future_prediction = grid.predict([[age_input, bmi_input, children_input, sex_male_input, smoker_yes_input]])
print("Future Prediction = {}".format(Future_prediction))


# In[ ]:




