#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np


# In[14]:


im_1=plt.imread('canakkale.jpg')


# In[15]:


im_1.shape


# In[16]:


im_2=np.zeros((356, 885),dtype=np.uint8)
im_2.shape


# In[17]:


im_2=im_1[:,:,0]  #filter uygulanacak resim


# In[18]:


im_3=im_1[:,:,0] # filter uygulandıktan sonrakı resim
im_3=np.zeros((356,885),dtype=np.uint8)


# In[19]:


plt.imshow(im_2,cmap='gray')
plt.show()


# In[20]:


m,n=im_2.shape


# In[21]:


for i in range(1,m-1):
    for j in range(1,n-1):
        s=         im_2[i-1,j-1]/9+         im_2[i-1,j]/9+         im_2[i-1,j+1]/9+         im_2[i,j-1]/9+         im_2[i,j]/9+         im_2[i,j+1]/9+         im_2[i+1,j-1]/9+         im_2[i+1,j]/9+         im_2[i+1,j+1]/9
        s=int(s)
        #print(s, end=' * ')
        im_3[i,j]=s


# In[22]:


plt.imshow(im_3,cmap='gray')
plt.show()


# In[23]:


plt.subplot(1,2,1)
plt.imshow(im_2,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(im_3,cmap='gray')

