#!/usr/bin/env python
# coding: utf-8

# # การบ้านงาน Lab 1 Basic Python
# 
# ### นายอธิศ สุนทโรดม
# ### Sec.2  65543206086-2

# ## แบบฝึกหัดข้อที่ 1

# In[4]:


text = input("ป้อนข้อความ : ")
word = []

words = text.split()

for x in words:
    if x not in ["the","a"]:
        i = len(x)
        word.append(i)

print("ความยาวของแต่ละคำ : ", word)


# In[5]:


text = input("ป้อนข้อความ : ")
word = []

words = text.split()

for x in words:
    if x not in ["the","a"]:
        i = len(x)
        word.append(i)

print("ความยาวของแต่ละคำ : ", word)


# ## แบบฝึกหัดข้อที่ 2

# ### 2.1

# In[6]:


import random

for i in range(10):
    x = ["H","T"]
    print(i,"ค่าที่ได้คือ :",random.choice(x))


# ### 2.2

# In[7]:


import random

x = ["H","T"]
tosses=[]
num_tosses = 100
for i in range(num_tosses):
    tosses.append(random.choice(x))
print(tosses[:])


# ### 2.3

# In[11]:


import random

x = ["H","T"]
tosses=[]
def toss_fair_coin(num_tosses):
    for i in range(num_tosses):
        tosses.append(random.choice(x))
    print(tosses[:])
    
toss_fair_coin(1000)


# ### 2.4

# In[12]:


x = tosses.count("H")
print("H =",x)


# ### 2.5

# In[13]:


import matplotlib.pyplot as plt
plt.hist(tosses)
plt.show()


# In[ ]:




