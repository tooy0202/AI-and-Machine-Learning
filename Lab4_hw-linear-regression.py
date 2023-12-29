#!/usr/bin/env python
# coding: utf-8

# # ใบงาน hw-linear-regression
# ### นายอธิศ สุนทโรดม
# ### Sec.2  65543206086-2

# In[55]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# 2.4 อ่านข้อมูลจากไฟล์ train.csv เก็บไว้ในตัวแปร df_train ด้วยคำสั่ง read_csv() จากไลบรารี pandas
df_train = pd.read_csv('train.csv')

df_train


# In[70]:


# 2.5  ทำการสำรวจข้อมูลในเบื้องต้น (data exploration)
# 2.5.1 ใช้คำสั่ง df_train.describe() เพื่อดูสถิติข้อมูลในแต่ละคอลัมน์
df_train.describe()


# In[71]:


min_max_values = df_train.describe().loc[['min', 'max']]
print(min_max_values )


# In[72]:


# 2.5.2 ใช้คำสั่ง df_train.corr() เพื่อหา correlation matrix ระหว่างแต่ละคู่ฟีเจอร์
numeric_columns = df_train.corr(numeric_only=True)
numeric_columns['SalePrice']


# In[73]:


# 2.5.3 ใช้คำสั่ง heatmap() จากไลบรารีseaborn เพื่อนำ correlation matrix ไปแสดงเป็นรูปภาพ
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=1, yticklabels=1)
plt.show()


# In[59]:


# ใช้คำสั่ง pairplot() จากไลบรารี seaborn ในการวาดกราฟ pair plot ระหว่าง SalePrice กับฟีเจอร์ 3 ฟีเจอร์จากข้อก่อนหน้าที่มีcorrelation กับ SalePrice สูงที่สุด
selected_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(df_train[selected_features], height=2)
plt.show()


# In[60]:


# 2.6  ทำการฝึกแบบจำลอง Linear Regression ทำนายค่าของ SalePrice จาก GrLivArea ของบ้าน โดยใช้ Train Linear Regression model
# 2.6.1 Create and train the model
model = LinearRegression()


# In[74]:


# 2.6.2 เตรียมข้อมูล X, y โดยนำข้อมูลจากคอลัมน์ GrLivArea และ SalePrice มาใส่ใน column vector ด้วยคำสั่ง numpy.c_[ ]
X = np.c_[df_train['GrLivArea']] # Feature
y = np.c_[df_train['SalePrice']]


# In[75]:


# 2.6.3 แบ่งข้อมูลจากข้อ 2.6.2 ออกเป็นชุด train และชุด test ในอัตราส่วน 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


# 2.6.4 ทำการเทรนแบบจำลองด้วยข้อมูลชุด train (80%)
model.fit(X_train, y_train)


# In[85]:


# 2.7  ทำการทดสอบแบบจำลองที่ฝึกแล้ว
# 2.7.1 นำแบบจำลองที่ฝึกแล้ว ไปใช้ predict กับข้อมูลชุด train (80%)
y_train_pred = model.predict(X_train)

# ใช้คำสั่ง mean_squared_error() ในการคำนวณหา RMSE
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

# Q3 รายงานค่า RMSE ของข้อมูลชุด train
print("RMSE ของข้อมูลชุด train:", rmse_train)


# In[86]:


# 2.7.2 นำแบบจำลองที่ฝึกแล้ว ไปใช้ predict กับข้อมูลชุด test (20%)
y_test_pred = model.predict(X_test)

# ใช้คำสั่ง mean_squared_error() ในการคำนวณหา RMSE
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

# Q4 รายงานค่า RMSE ของข้อมูลชุด test
print("RMSE ของข้อมูลชุด test:", rmse_test)


# In[79]:


# 2.8.1 วาดกราฟของจุดข้อมูลชุด train ทั้งหมด
plt.scatter(X_train, y_train, alpha=0.5)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()


# In[80]:


# 2.8.2 วาดกราฟ hyperplane
plt.scatter(X_train, y_train, alpha=0.5, label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Linear Regression')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.legend()
plt.show()


# # 3 ตอบคำถาม
# 
# **สร้าง Markdown cell ตอบคำถาม Q1 – Q4 จากข้อ 2 โดยให้ตอบใน cell เดียวกันไว้ข้างล่างสุดของไฟล์แยกจากส่วนของโค้ดในข้อ 2**
# 
# - **Q1 :** สำรวจจาก count ว่ามีคอลัมน์ไหนที่มีข้อมูลสูญหาย (missing data) บ้าง (สามารถใช้ .iloc[0] เพื่อเลือกดูเฉพาะ row count สำหรับทุกคอลัมน์)
#   

# In[68]:


#Q1 ข้อมูลที่สูญหาย
for i in missing_count :
    print(missing_count)


# - **Q2 :** 
#   สังเกตใน row ของ SalePrice ว่าฟีเจอร์ใดมีค่า correlation กับ SalePrice สูงสุดเป็นอันดับต้นๆ บ้าง
#   : 'OverallQual', 'GrLivArea', 'GarageCars'
# - **Q3 :** 
#   รายงานค่า RMSE ของข้อมูลชุด train : 55480.7719291839
# - **Q4 :** 
#    รายงานค่า RMSE ของข้อมูลชุด test :  58471.75652552954

# In[ ]:




