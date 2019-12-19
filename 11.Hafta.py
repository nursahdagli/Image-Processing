#!/usr/bin/env python
# coding: utf-8

# In[3]:


#06.12.19 dersi
#pikselin ağırlıgını ölçmeden hassasiyeti hesaplama
import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=5, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        sayac=0
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                sayac+=1
        print("döngü sayısı : ", sayac)


# In[4]:


import numpy as np
#from perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0]) #and
#labels = np.array([1, 1, 1, 0]) #or
#labels = np.array([0, 1, 1, 0]) #xor

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 1])
perceptron.predict(inputs) 
#=> 1

inputs = np.array([0, 1]) #1,1 yazdıgında 1 çıktısı al
perceptron.predict(inputs) #tahmin
#=> 0


# In[5]:


perceptron.weights


# In[6]:


perceptron.threshold

