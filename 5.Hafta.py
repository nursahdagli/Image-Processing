#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


data_path ="C:\\Users\\Nurşah DAĞLI"
train_data = np.loadtxt(data_path + "\mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "\mnist_test.csv", 
                       delimiter=",")


# In[6]:


image_size = 28 # width and length
no_of_different_labels = 10 # 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size #784
data_path = "data/mnist/"
test_data[:10] #tüm satırları ve 10. sutuna kadar gösterir.


# In[7]:


train_data.ndim, train_data.shape


# In[8]:


train_data[10,0] #10.satır 0. eleman


# In[32]:


im_3=train_data[10,:] #im_3 10.satırdaki tüm sütunlar boyutunda


# In[11]:


im_3.shape


# In[12]:


im_4=im_3[1:] #ilk satır sayıları gösterdiği için çıkartıyoruz.


# In[13]:


im_4.shape


# In[16]:


im_5=im_4.reshape(28,28) #im_5 im_4'un 28*28 lik hali 


# In[34]:


m,n=train_data.shape
m,n


# In[36]:


plt.imshow(im_5,cmap="gray")
plt.show()


# In[19]:


60000 ,785 ; 1+ 28*28


# In[20]:


#train datada rakamlardan kaçar tane oldugunu bulan fonksiyon
def my_counter(k=0): 
    s=0
    for i in range(m):
        if(train_data[i,0]==k):
            s=s+1
    return s
for i in range(10): 
    c=my_counter(i)
    print(i," ",c)


# In[43]:


#rakamların bulunma olasılıgını hesaplayan fonksiyon
import  math
def my_pdf_1(x, mu=0.0 , sigma=1.0):
    x= float(x -mu) / sigma
    return math.exp(-x*x/2.0) / math.sqrt(2.0*math.pi) / sigma
my_pdf_1(10,1,3)


# In[22]:


def get_my_mean_and_std(k=0,l=350):
    s=0 #kactane sıfır var onu saysın//kac digit oldugu
    t=0 #intersitiy degeri pixeldeki
    #k=0 # sınfı bilgisi yani digitin
    #l=350  #location ı belirtiyor.classın pixel degeri
    for i in range(m):  #ortalamayı buldurdu
        if(train_data[i,0]==k):
            s=s+1
            t=t+train_data[i,l+1]
           # digit_class=train_data[i,0]
            #top_left=train_data[i,1]
            #bottom_right=train_data[i,784]
           # print(digit_class,end=" ")
            #print(top_left,end=" ")
          #  print(bottom_right,end=" \n")      
    mean_1=t/s

    s=0
    t=0
    for i in range(m):
        if(train_data[i,0]==k):
            s=s+1
            diff_1=train_data[i,l+1]-mean_1
            t=t+diff_1*diff_1
    #var_1=t/(s-1)
    std_1=np.sqrt(t/(s-1))

    print(mean_1,std_1)
    return mean_1,std_1
        # train_data[i,0] #label
        # train_data[i,1] #sol üstteki deger
        # train_data[i,784]#en alt kosedeki deger 


# In[45]:


m_1,std_1=get_my_mean_and_std(2,100)
my_pdf_1(40,m_1,std_1)


# In[31]:


im_1=plt.imread("iki.png")
plt.imshow(im_1)
plt.show()
test_value=im_1[0,0,0]


# In[ ]:




