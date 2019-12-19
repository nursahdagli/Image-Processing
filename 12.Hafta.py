#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_mldata


# In[ ]:


mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]


# In[ ]:


type(mnist)


# In[ ]:


import numpy as np

y_new = np.zeros(y.shape)
y_new[np.where(y == 0.0)[0]] = 1
y = y_new


# In[ ]:


m = 60000
m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T
y_train, y_test = y[:m].reshape(1,m), y[m:].reshape(1,m_test)


# In[ ]:


np.random.seed(138)
shuffle_index = np.random.permutation(m)
X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index] #veriyi karıştırıyor


# In[ ]:


get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

i = 3
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(y_train[:,i])


# In[ ]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# In[ ]:


def compute_loss(Y, Y_hat): #gerçek çıktı, benim ürettiğim çıktı iki veri alıyor

    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )

    return L


# In[ ]:


learning_rate = 1

X = X_train
Y = y_train

n_x = X.shape[0]
m = X.shape[1]

W = np.random.randn(n_x, 1) * 0.01
b = np.zeros((1, 1))

for i in range(100): #daha hızlı çalışması için range 2000 den 100 düşürdük
    Z = np.matmul(W.T, X) + b
    A = sigmoid(Z)

    cost = compute_loss(Y, A)

    dW = (1/m) * np.matmul(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)

    W = W - learning_rate * dW #eski degerleri yeni degerlere atadı
    b = b - learning_rate * db

    if (i % 5 == 0): #her 5 adımda bir rapor sun. her bir adımda hata azalmalıdır
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix

Z = np.matmul(W.T, X_test) + b
A = sigmoid(Z)

predictions = (A>.5)[0,:]
labels = (y_test == 1)[0,:]

print(confusion_matrix(predictions, labels)) #dogru ve yanlış class bilgisini sunuyor 

#burda bir matris olustu ve doğru class ve yanlıs class bilgisini verdi, ikisinin toplamı dogru sonucu vermesi gerekiyor
#confision matris olarak geçiyor.
  # A     B
# A 9000  70
# B 40    800 BURDA 9000 ve 800 doğru degerler oldu. Bir şekilde performans ölçüldü.


# In[ ]:


print(classification_report(predictions, labels))
# yine hata ölçüm yöntemleri

# In[ ]:

#Bir One Hidden Layer Ekledik

X = X_train
Y = y_train

n_x = X.shape[0]
n_h = 64
learning_rate = 1

W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(1, n_h)
b2 = np.zeros((1, 1))

for i in range(50):

    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    cost = compute_loss(Y, A2)

    dZ2 = A2-Y
    dW2 = (1./m) * np.matmul(dZ2, A1.T)
    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1./m) * np.matmul(dZ1, X.T)
    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if i % 10 == 0:
        print("Epoch", i, "cost: ", cost)

print("Final cost:", cost)

