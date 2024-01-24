#!/usr/bin/env python
# coding: utf-8

# # Lab5 logistic-regression
# ### จัดทำโดย นาย อธิศ สุนทโรดม
# ### รหัส 65543206086-2

# In[17]:


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#print(iris.DESCR)
#iris.data.shape
X_train = iris.data[:,2:] # 2คอลัมน์ petal Length, petal wisth
y_train = (iris.target == 2) #เป็นพันธุ์ virgi หรือไม่ (binary 0,1)

logre = LogisticRegression()
logre.fit(X_train,y_train)

# มิติแรก : ทั้ง150ดอก
# มิติสอง 0 = คอลัมน์ petal len, 1=คอลัมน์ petal width
#plt.figure(figsize=(3,2))
plt.scatter(X_train[(y_train == 1),0],X_train[(y_train == 1),1]) #วาดข้อมูล virginica
plt.scatter(X_train[(y_train == 0),0],X_train[(y_train == 0),1]) #วาดข้อมู not virginica
plt.xlabel('Petal Length (X1)')
plt.ylabel('Petal Width (X2)]')

import numpy as np
theta0 = logre.intercept_
theta1 = logre.coef_[0,0]
theta2 = logre.coef_[0,1]
x1 = np.arange(3,8)
x2 = (-theta0-theta1*x1)/theta2 #สมการ decision  boundary (p_hat = 0.5)
plt.plot(x1,x2,'r')


# In[18]:


#ใช้ model predict ดอกใหม่ๆ
X_test = [[6,2],[4,1],[2.3, 0.8],[7,3],[5,1.8],[4.9,1.8],[4.991,1.53425007]]
print(logre.predict(X_test)) # True = Virginica
print(logre.predict_proba(X_test)) # P(not virgi), P(virgi)


# In[19]:


iris.target


# In[20]:


iris.data[0]


# In[21]:


iris.data[0,2]


# In[ ]:




