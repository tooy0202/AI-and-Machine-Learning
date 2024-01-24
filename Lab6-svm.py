#!/usr/bin/env python
# coding: utf-8

# # Lab6 svm
# ### จัดทำโดย นาย อธิศ สุนทโรดม
# ### รหัส 65543206086-2

# In[8]:


import pandas as pd
import numpy as np

df = pd.read_csv('diabetes.csv')
df.describe()


# In[2]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('diabetes.csv')
x = df.drop(columns=['Outcome'])
y = df['Outcome']

# clean ข้อมูลที่ผิดปกติ (เช่น glucose = 0)
# mean imputation (เติมค่าที่หายด้วย mean ของคอลัมน์นั้น)
cols = ['Glucose','BloodPressure','SkinThickness', 'Insulin','BMI']
for c in cols:
    x[c] = x[c].replace(0,np.NaN) #เพื่อให้การคำนวณ mean ไม่รวมค่า 0 
    mean = x[c].mean()
    x[c] = x[c].replace(np.NaN,mean) #แทนค่าที่หายด้วย mean


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

svm = Pipeline([
    ('scaler',StandardScaler()),
    ('svc',LinearSVC(C=100,max_iter = 50000))
])


svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
acc = np.sum(svm.predict(x_test) == y_test)/len(x_test)
acc

# acc (ก่อน clean ข้อมูล): 0.72-0.77
# acc (หลัง clean):0.66-0.79


# In[26]:


# mean imputation (เติมค่าที่หายด้วย mean ของคอลัมน์นั้น)
import numpy as np
df['Glucose'] = df['Glucose'].replace(0,np.NaN) #เพื่อให้การคำนวณ mean ไม่รวมค่า 0 
mean = df['Glucose'].mean()

df['Glucose'] = df['Glucose'].replace(np.NaN,mean) #แทนค่าที่หายด้วย mean


# In[4]:


# เปลี่ยนมาใช้ svm แบบมี kernel (linearSVC => SVC)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = Pipeline([
    ('scaler',StandardScaler()),
    ('svc',SVC(kernel='rbf',max_iter = 50000))
])
hyperparam_grid = [{
    'svc__C':[1,10,100,1000,10000],
    'svc__kernel':['rbf','linear']
}]
clf = GridSearchCV(svm,hyperparam_grid,cv=5,scoring='accuracy')
clf.fit(x_train,y_train)
clf.cv_results_
#y_pred = svm.predict(x_test)
#acc = np.sum(svm.predict(x_test) == y_test)/len(x_test)
#acc


# In[5]:


clf.best_params_


# In[6]:


y_pred = clf.predict(x_test)
acc = np.sum(y_pred == y_test)/len(x_test)
acc


# In[ ]:




