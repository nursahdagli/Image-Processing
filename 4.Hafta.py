#!/usr/bin/env python
# coding: utf-8

# In[12]:


#standart sapma ve ortalama hesaplama fonksiyonu
import math
def my_f_1(my_list=[2,4,3,40,5,6,3,3,2,1]):
    orttoplam=0
    stdtoplam=0
    for i in my_list:
        orttoplam+=i
        
    mean=orttoplam/len(my_list) #ortalama

    for i in my_list: #standart sapma hesaplama
        stdtoplam+=(i-mean)*(i-mean)

    var=stdtoplam/(len(my_list)-1)
    var=math.sqrt(var)
    
    return mean,var

print(my_f_1())


# In[14]:


#dizideki elemanların kaç defa tekrar ettiğini my_histogram listesinde key:value şeklinde tutuyor
my_histogram={}
my_list=[2,4,3,40,5,6,3,3,2,1]

for i in my_list:
    if i in my_histogram.keys():
        my_histogram[i]+=1
    else:
        my_histogram[i]=1 #dizide her eleman 1 defa oldugu için.

#my_histogram[1]=10 #value:1 değerinde olanın key:10 yap 
#my_histogram[2]=15
#my_histogram[40]=40
print(my_histogram)


# In[24]:


#resimdeki

import matplotlib.pyplot as plt
import numpy as np

def my_f_2(image_1=plt.imread('istanbul.jpg')):
    print(image_1.ndim,image_1.shape)
    m,n,p=image_1.shape
    my_histogram={}
    for i in range(m):
        for j in range(n):
            if (image_1[i,j,0] in my_histogram.keys()):
                my_histogram[image_1[i,j,0]]+=1
            else:
                my_histogram[image_1[i,j,0]]=0
            #image_1[i,j,1]
            #image_1[i,j,2]

    return my_histogram

print(my_f_2())


# In[28]:


x=[]
y=[]
my_histogram=my_f_2()
for key in my_histogram.keys():
    x.append(key)
    y.append(my_histogram[key]) 
plt.bar(x,y)
plt.show()


# In[ ]:




