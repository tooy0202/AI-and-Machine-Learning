#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn import datasets
digits = datasets.load_digits()
digits


# In[16]:


print(digits.DESCR)


# In[17]:


digits.keys()


# In[18]:


digits.data
digits.images
digits.target


# In[19]:


digits.images.shape


# In[20]:


digits.images[0]


# In[21]:


import matplotlib.pyplot as plt
plt.figure(figsize=(2,2))
im = digits.images[1600]
label = digits.target[1600]
print(label)
plt.imshow(im)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
clf =KNeighborsClassifier()
x_train = digits.data[:1500]
y_train =  digits.target[:1500]
X_test = digits.data[1500:]
y_test = digits.target[1500:]
#
clf.fit(x_train,y_train) #train (X feature,y label target)
y_pred = clf.predict(X_test) #infornce
#
n_correct = sum(y_pred == y_test)
accuracy = n_correct/len(y_test)
print(accuracy)
print('ตอบถูก', n_correct,'จากทั้งหมด', len(y_test))


# In[25]:


#y_pred == y_test ผิดที่ตัว 54
print(y_pred[53])
print(y_test[53])
plt.figure(figsize=(2,2))
plt.imshow(digits.images[1553])


# In[27]:


#วาด confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(2,2))
plt.imshow(cm)


# In[28]:


cm


# In[ ]:




