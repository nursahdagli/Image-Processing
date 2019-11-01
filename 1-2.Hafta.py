#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
os.getcwd() #dosya yolunu gösterir.
os.listdir() #o dizindeki dosyaları listeler.


# In[6]:


jpg_files=[f for f in os.listdir() if f.endswith(".jpg")]
print(jpg_files) #dizinde jpg varsa yazdırır.


# In[7]:


im_1=plt.imread('canakkale.jpg')
print(type(im_1))


# In[8]:


print(im_1.ndim) #kac boyutlu oldugu
print(im_1.shape) #boyutun piksel değerleri


# In[9]:


print(im_1[100,100,:])


# In[13]:


m,n,p=im_1.shape


# In[10]:


plt.imshow(im_1)
plt.show()


# In[11]:


im_2=im_1[:,:,0] #im_1 resmini iki boyuta indirdi.
plt.imshow(im_2)
plt.show()
plt.imsave('MerhabaSonSınıf.jpg',im_2) #verilen isimle resmi kaydetti.


# In[18]:


#resmi siyah-beyaz yapma
new_image=np.zeros((m,n),dtype=float) #im_1 resmi ile aynı boyutlarda yeni resim oluşturdu.
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,0]+im_1[i,j,1]+im_1[i,j,2])/3 #rgb değerlerinin ortalamasını alıyor.
        new_image[i,j]=s
plt.imshow(new_image,cmap='gray')
plt.show()
plt.imsave('test1.jpg',new_image,cmap='gray')


# In[19]:


#2 boyutlu resmi siyah-beyaz yapma
new_image=np.zeros((m,n),dtype=float) #im_1 resmi ile aynı boyutlarda yeni resim oluşturdu.""
for i in range(m):
    for j in range(n):
        s=(im_2[i,j]) 
        new_image[i,j]=s
plt.imshow(new_image,cmap='gray')
plt.show()
plt.imsave('test1.jpg',new_image,cmap='gray')


# In[25]:


#resmi 90 derece döndürme
new_image=np.zeros((n,m),dtype=float) #m,n değerlerini değiştirdik.
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,0]+im_1[i,j,1]+im_1[i,j,2])/3
        new_image[j,i]=s #i,j değerlerini değiştirdik.
plt.imshow(new_image,cmap='gray')
plt.show()
plt.imsave('test1.jpg',new_image,cmap='gray')


# In[ ]:


#renk ters cevirme fonksiyonu
def my_inverse(im_1):
    return(255-im_1)


# In[26]:


im_3=im_1+25 #parlalıklık arttırır.
im_4=255-im_1 #renk ters cevirir.
plt.imshow(im_3)
plt.show()
plt.imshow(im_4)
plt.show()


# In[ ]:




