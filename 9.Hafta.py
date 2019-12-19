#!/usr/bin/env python
# coding: utf-8

# In[1]:


v_1=[1,3]
v_2=[2,-3]
v_1,v_2


# In[2]:


v_1+v_2  # iki vektörü birleştiriyor ( [1, 3, 2, -3] )


# In[3]:


def my_add(vector_1,vector_2):     # uzunluğu bilinen
    vector_3=[0,0]
    vector_3[0]=vector_1[0]+vector_2[0] #ilk elemanları topluyor
    vector_3[1]=vector_1[1]+vector_2[1] #ikinci elemanları topluyor
    
    return vector_3 # [3,0]

def my_add_1(vec_1,vec_2):  #uzunluğu bilinmeyen
    
    s=len(vec_1)
    result_vec=[]
    
    for i in range (s):
        temp=vec_1[i]+vec_2[i]
        result_vec.append(temp)
    
    return result_vec # [3,0]

import random
def my_create_vectors(m=5,n=2):
    m,n=2,3 #iki tane 3 boyutlu liste
    my_vec=[]
    for i in range(m):
        my_vec.append([])
        for j in range(n):
            t=random.randint(-10,10)
            my_vec[i].append(t) #ilk önce ilk vektörü daha sonra ikinci vektörü oluşturuyor.
            #pass
    return my_vec        


# In[4]:


my_create_vectors()


# In[5]:


my_add(v_1,v_2)


# In[6]:


my_add_1(v_1,v_2)


# In[7]:


v_1=[1,3,1]
v_2=[2,-3,6]
my_add_1(v_1,v_2)


# In[8]:


def my_center(vec_1,vec_2): #vektörlerin her bir elemanları için ortalamalarını buluyor
    
    s=len(vec_1)
    result_vec=[]
    
    for i in range (s):
        temp=(vec_1[i]+vec_2[i])/2
        result_vec.append(temp)
    return result_vec


# In[9]:


v_1=[1,3,1]
v_2=[-2,-3,6]
my_center(v_1,v_2)


# In[10]:


def my_distance(v_1,v_2): #vektörler arası uzaklığı hesaplıyor
    s=len(v_1)
    t=0
    for i in range(s):
        t=t+(v_1[i]-v_2[i])**2
    return t**0.5


# In[11]:


vector_1=[0,4]
vector_2=[3,0]
my_distance(vector_1,vector_2)


# In[12]:


vector_3=[0,4,0]
vector_4=[3,0,0]
my_distance(vector_1,vector_2)

