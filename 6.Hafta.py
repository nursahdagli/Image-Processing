#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data_path ="C:\\Users\\Nurşah DAĞLI"
# train_data = np.loadtxt(data_path + "\mnist_train.csv", 
#                         delimiter=",")
test_data = np.loadtxt(data_path + "\mnist_test.csv", 
                       delimiter=",")


# In[3]:


eps=np.finfo(float).eps               #rakamların bulunma olasılığı fonksiyonu
import  math
def my_pdf_1(x, mu=0.0 , sigma=1.0):
    
    x= float(x -mu) /sigma
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) /sigma


# In[4]:


def get_my_mean_and_std(k=0,l=350):
    s=0 # kactane sıfır var onu saysın//kac digit oldugu
    #k=0 # sınfı bilgisi yani digitin
    t=0 #intersitiy degeri pixeldeki
    #l=350  #location ı belirtiyor.classın pixel degeri
    for i in range(10000):  #m train olursa 60000 test olursa 10000
        if(test_data[i,0]==k):
            s=s+1
            t=t+test_data[i,l+1]
           # digit_class=train_data[i,0]
            #top_left=train_data[i,1]
            #bottom_right=train_data[i,784]
           # print(digit_class,end=" ")
            #print(top_left,end=" ")
          #  print(bottom_right,end=" \n")      
    mean_1=t/s

    s=0
    t=0
    for i in range(10000):
        if(test_data[i,0]==k):
            s=s+1
            diff_1=test_data[i,l+1]-mean_1
            t=t+diff_1*diff_1
    #var_1=t/(s-1)
    std_1=np.sqrt(t/(s-1))

    #print(mean_1,std_1)
    return mean_1,std_1
        # train_data[i,0] #label
        # train_data[i,1] #sol üstteki deger
        # train_data[i,784]#en alt kosedeki deger 


# In[5]:


get_my_mean_and_std(1,10)


# In[6]:


test_data[100,:]  #test_data'nın 100. satırdaki tüm sütunların değeri


# In[15]:


my_pdf_1(45.8,40,20) 


# In[17]:


im_1=plt.imread('resim.jpg')
plt.imshow(im_1)
plt.show()


# In[8]:


im_1.shape


# In[9]:


im_2=im_1[:,:,0] #im_1 i iki boyutlu hale getirmek için


# In[10]:


im_2.shape


# In[11]:


im_1[14,:]


# In[12]:


im_5=im_2.reshape(1,784) #im_2 resmini düzleştirdi.


# In[13]:


im_5.shape


# In[14]:


liste = list()   #pdf değerlerini tutmak için
for i in range(10):
    pdf_t=0
    for j in range(784): #resmin boyutu
        
        x=im_5[0,j]
        m_1,std_1=get_my_mean_and_std(i,j) #butun piksellerin ortalaması ve varyansını buluyor.
        pdf_deger=my_pdf_1(x,m_1,std_1+eps)
        pdf_t+=pdf_deger
        liste.append(pdf_t) #pdf degerlerini listeye atıyor
m=len(liste)
maxNumber=0
for i in range(m): #listedeki en büyük pdf degerini bulmak için
    if maxNumber < liste[i]:
        maxNumber = liste[i]
print(maxNumber)


# In[ ]:




