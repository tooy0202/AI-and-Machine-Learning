#!/usr/bin/env python
# coding: utf-8

# In[74]:


import cv2
import matplotlib.pyplot as plt

im = cv2.imread('twice.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_blur = cv2.GaussianBlur(im,(311,31),0)
edges = cv2.Canny(im,0,50)

plt.imshow(im_blur)
plt.imshow(edges)


# In[38]:


plt.imshow(im[:,:,1],cmap='gray')


# In[95]:


import pandas as pd

df = pd.read_csv('Pokemon.csv')
df.sort_values(['Defense'],ascending=False)


# In[93]:


df['hpatk'] = df['HP'] + df['Attack']


# In[94]:


df = df.drop(columns=['hpatk'])


# In[97]:


df.groupby('Type 1').mean('Attack')


# In[ ]:




