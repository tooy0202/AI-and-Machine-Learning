#!/usr/bin/env python
# coding: utf-8

# # ใบงาน 3-kNN
# ### นายอธิศ สุนทโรดม
# ### Sec.2  65543206086-2

# # k-Nearest Neighbor (kNN) exercise
# 
# **kNN classifier ประกอบไปด้วย 2 ขั้นตอนหลัก คือ:**
# 
# - **ในขั้นการ train** 
#   จะทำเพียงเก็บ (จำ) ข้อมูล training set ทั้งหมดไว้ในตัวแปร 
#   
#   
# - **ในขั้นการ predict** อัลกอริทึม kNN ทำนายค่า label ของแต่ละ test image ตาม majority vote ของเพื่อนบ้าน โดยมี 3 ขั้นตอนย่อยดังนี้
#   1. คำนวณหาระยะห่าง (L2 distance) ระหว่าง test image และ train image ทุกคู่ภาพ 
#   2. สำหรับแต่ละ test image เราจะหาภาพ train image ที่มีค่า distance น้อยที่สุด k ภาพ (distance น้อย = ที่มีลักษณะใกล้เคียงที่สุด)
#   3. ตอบ label ตาม majority vote ของ k ภาพดังกล่าว 

# In[1]:


# โค้ด setup ต่างๆ
import random
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray_r'

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### จง implement ฟังก์ชันสำหรับ kNN ต่อไปนี้
# 
# 1. **train(X, y)** มีหน้าที่จำข้อมูล training set เก็บไว้ในตัวแปร
# 2. **predict(X, k)** มีหน้าที่ทำนาย label ของข้อมูลชุด test โดยใช้เพื่อนบ้านที่ใกล้ที่สุด k ตัว
# 3. **compute_distances(X)** มีหน้าที่คำนวณระยะห่างระหว่างทุกคู่ภาพ train และ test images โดยเราจะเก็บค่าระยะห่างทั้งหมดไว้ใน distance matrix เช่น สมมติว่าเรามี **1000** training images และมี **200** test images เราจะได้ distance matrix ที่มีขนาด **200 x 1000** โดยที่แต่ละค่าใน matrix ณ ตำแหน่ง (i,j) จะเก็บค่าระยะห่างระหว่าง i-th test และ j-th train image 
# 
# **หมายเหตุ**
# - การคำนวณ L2 distance เป็นตามสมการ $ d_2(I_1,I_2) = \lVert I_1 - I_2 \rVert_2 = \sqrt{\sum_p{(I_1^p - I_2^p)^2}} $

# In[41]:


X_train = []
y_train = []

def train(X, y):
    """
    จำข้อมูลชุด train ทั้งหมด เก็บไว้ในตัวแปร X_train, y_train
    """
    X_train = X
    y_train = y
    
    
def predict(X, k=1):
    """
    ทำนาย label ของข้อมูลชุด test

    Inputs:
    - X: ข้อมูลชุด test
    - k: ค่า hyperparameter k ของอัลกอริทึม k-NN

    Returns:
    - y: numpy array ของ label ที่ทำนาย โดยให้ y[i] เก็บค่า label ของภาพที่ i  
    """

    dists = compute_distances(X)
    num_test = X.shape[0]
    y_pred = np.zeros(num_test)
    
    for i in range(num_test):
        # อาเรย์ขนาด k ซึ่งจะเก็บ k nearest neighbors ของข้อมูลชุด test ตัวที่ i
        closest_y = []
        
        # TODO: ใช้ distance matrix ในการหา k nearest neighbors
        # และใช้ y_train ในการหาค่า label เพื่อนบ้าน k ตัวดังกล่าว
        # เก็บคำตอบไว้ใน closest_y
        # hint: สามารถใช้ฟังก์ชัน numpy.argsort เพื่อหา index ของ distance ที่น้อยที่สุด k ค่า
        closest_y = y_train[np.argsort(dists[i])[:k]]
        
        
        # TODO: หา majority vote ของเพื่อนบ้าน k ตัวแล้วเก็บผลลัพธ์ใน y_pred
        # หากเกิดการเสมอกัน ให้เลือก label ที่มีค่าที่น้อยกว่า
        # hint: ในการนับ vote สามารถใช้ฟังก์ชัน numpy.argmax ร่วมกับ numpy.bincount ได้ หรือไม่ก็เขียนด้วย for-loop
        y_pred[i] = np.argmax(np.bincount(closest_y))
        
    return y_pred

def compute_distances(X):
    """
    คำนวณระยะห่างระหว่างข้อมูลชุดทดสอบทุกตัว กับข้อมูลชุดฝึกทุกตัว

    Inputs:
    - X: ข้อมูลชุดทดสอบ

    Returns:
    - dists: distance matrix ขนาด (num_test, num_train) ซึ่ง dists[i, j]
    จะเก็บค่าระยะห่างแบบ L2 ระหว่าง test image ภาพที่ i กับ training image ภาพที่ j
    """
    num_test = X.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    #for i in range(num_test):
        #for j in range(num_train):
            # TODO: คำนวณหา dists[i,j] โดยพยายามไม่ใช้ loop ซ้อนอีก
            # hint: มีคำสั่ง np.sqrt(), np.sum(), np.square(), np.linalg.norm() ที่ช่วยคำนวณได้
            
           
           # pass
    dists = np.sqrt(np.sum(np.square(X[:, np.newaxis] - X_train), axis=2))
    return dists


# ### จงนำฟังก์ชัน train, predict ที่ implement เรียบร้อยแล้วไปใช้จำแนกภาพกับข้อมูล MNIST และ CIFAR-10
# 
# #### 1. เริ่มจากทำการโหลดข้อมูลภาพและ label แบ่งออกเป็นชุด train/test

# In[7]:


# โหลด MNIST dataset จาก sklearn.dataset เก็บไว้ในตัวแปร digits
from sklearn import datasets
digits = datasets.load_digits()

# TODO: ตัดแบ่ง images และ target ออกเป็นส่วนชุด train และชุด test
# ให้ชุด train มีขนาด 1500 ภาพ 
# ให้ชุด test มีขนาด 297 ภาพ
X_train = digits.data[:1500]
y_train = digits.target[:1500]
X_test = digits.data[1500:]
y_test = digits.target[1500:] 

# ตรวจสอบความถูกต้องของขนาดข้อมูล
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# visualize ตัวอย่างภาพจากแต่ละ class โดยสุ่ม class ละ 5 ภาพ
classes = [0,1,2,3,4,5,6,7,8,9]
num_classes = len(classes)
samples_per_class = 5
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].reshape(8, 8).astype('uint8'), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


# #### 2. ก่อนนำภาพไปใช้ train/test จะต้อง flatten ภาพจากขนาด 8 x 8 ให้กลายเป็นเวกเตอร์ขนาด 64

# In[4]:


X_train = X_train.reshape(-1, 64)
X_test = X_test.reshape(-1, 64)


# #### 3. เรียกใช้ฟังก์ชัน train และ predict 

# In[39]:


# TODO: ทดลองเปลี่ยนค่า k
train(X_train, y_train) #เรียกใช้การเทน
ypred = predict(X_test, k=3) #คำตอบที่ทำการเทนเรียบร้อยโดนมี k = 3


# #### 4. คำนวณหา accuracy = อัตราส่วนของ label ที่ทำนายถูก
# 
# $ accuracy = \frac{\mbox{number of correct labels}}{\mbox{total number of predictions}} $

# In[40]:


# TODO: คำนวณค่าเหล่านี้
num_correct = sum(ypred == y_test)
num_test = X_test.shape[0]
accuracy = num_correct/num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# #### (เสริม) ทดสอบความถูกต้องของ compute_distances
# 
# เราสามารถนำ distance matrix ที่ได้จากมาวาดด้วย `plt.imshow()` ซึ่ง
# - แต่ละ row คือ test example และ แต่ละ column คือ train example 
# - สีขาว <-> ดำ บ่งบอกถึงระยะห่างที่มาก <-> น้อย ระหว่าง train/test example คู่นั้น

# In[14]:


dists = compute_distances(X_test)
plt.imshow(dists, interpolation='none',cmap='gray')#
plt.show()


# **Question #1:** สังเกตว่าในภาพของ distance matrix จะมี pattern บาง row จะมีสีดำเป็นพิเศษ จงตอบคำถาม
# 
# - แถบ row ที่มีสีดำเกิดจากอะไรในข้อมูล จงอธิบายตามความเข้าใจ
# - แถบ columns ที่มีสีดำเกิดจากอะไรในข้อมูล จงอธิบายตามความเข้าใจ

# **ตอบ**: แถบ row ที่มีสีดำเป็นพิเศษ และแถบ columns ที่มีสีดำเป็นพิเศษ เกิดจาก test example และ train example ที่มีลักษณะคล้ายกันจำนวนมาก ซึ่งส่งผลให้ระยะ   
#                  ห่างระหว่างทั้งสองมีน้อย

# **Question #2:** จงทดลองเปลี่ยนค่า k ตั้งแต่ 1..10 แล้วหาว่าค่า k ใดที่ทำให้ accuracy สูงที่สุด

# **ตอบ**: 
# - k = 1 Got 281 / 297 correct => accuracy: 0.946128
# - k = 2 Got 284 / 297 correct => accuracy: 0.956229
# - k = 3 Got 285 / 297 correct => accuracy: 0.959596
# - k = 4 Got 285 / 297 correct => accuracy: 0.959596
# - k = 5 Got 284 / 297 correct => accuracy: 0.956229
# - k = 6 Got 281 / 297 correct => accuracy: 0.946128
# - k = 7 Got 281 / 297 correct => accuracy: 0.946128
# - k = 8 Got 280 / 297 correct => accuracy: 0.942761
# - k = 9 Got 280 / 297 correct => accuracy: 0.942761
# - k = 10 Got 280 / 297 correct => accuracy: 0.942761
# - สรุป k ที่มีค่า accuracy สูงที่สุดคือ 3 และ 4 คือ Got 285 / 297 correct => accuracy: 0.959596

# ### ทิ้งท้าย: อยากให้ test accuracy ของ kNN สูงขึ้น ต้องทำอย่างไร? 
# 
# - สังเกตุว่าเราสามารถปรับแต่งหรือจูน **hyperparameter** ของ kNN ได้ เช่น 
#     - เปลี่ยนวิธีคำนวณระยะห่าง (L1 vs L2)
#     - เปลี่ยนค่า k ที่ใช้
#     - เปลี่ยนการให้น้ำหนักเพื่อนบ้านเวลา vote
# - ในคาบเรียนสัปดาห์หน้าเราจะพบว่า
#     - หากเลือก hyperparameter ไม่ดี อาจก่อให้เกิดปัญหา overfitting/underfitting ซึ่งจะส่งผลให้ค่า test accuracy ต่ำลง
#     - เราสามารถใช้เทคนิค Grid Search ร่วมกับ Cross Validation ในการ tune ค่า hyperparameter เหล่านี้โดยอัตโนมัติได้
