#resim yükleme,gri yapma, ters çevirme

import numpy as np
import matplotlib.pyplot as plt
import os

jpg_files=[f for f in os.listdir() if f.endswith(".jpg")]
print(jpg_files)

im_1=plt.imread('canakkale.jpg')
print(type(im_1))
print(im_1.ndim)
print(im_1.shape)
print(im_1[100,100,:])
m,n,p=im_1.shape

#plt.imshow(im_1)
#plt.show()

#new_image=np.zeros((m,n),dtype=float)
#for i in range(m):
#    for j in range(n):
#        s=(im_1[i,j,0]+im_1[i,j,1]+im_1[i,j,2])/3
#        new_image[i,j]=s
#plt.imshow(new_image,cmap='gray')
#plt.show()
#plt.imsave('test1.jpg',new_image,cmap='gray')

new_image=np.zeros((n,m),dtype=float)
for i in range(m):
    for j in range(n):
        s=(im_1[i,j,0]+im_1[i,j,1]+im_1[i,j,2])/3
        new_image[j,i]=s
plt.imshow(new_image,cmap='gray')
plt.show()
plt.imsave('test1.jpg',new_image,cmap='gray')