import cv2 
import numpy as np
from matplotlib import pyplot as plt
img= cv2.imread('/home/user/resistor/test/330_resistor.jpg')

#hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#bilateralFilter(image, dstBila, kernel_length, kernel_length*2, kernel_length/2);
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


#edges = cv2.Canny(blur,100,200)
high_thres,thres_img=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
low_thres=0.5*high_thres
blur = cv2.bilateralFilter(rgb,9,high_thres,low_thres)
gaussian_blur=cv2.GaussianBlur(gray,(5,5),0)
edges1 = cv2.Canny(blur,100,200)
edges = cv2.Canny(gaussian_blur,100,200)

#===========Showing original and Blurred image=============
plt.subplot(121),plt.imshow(rgb),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('blurred')
plt.xticks([]), plt.yticks([])
plt.show()
#=====================
plt.subplot(121),plt.imshow(gray,cmap='gray'),plt.title('Gray')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges1,cmap='gray'),plt.title(' Bi lateral Edges')
plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(121),plt.imshow(edges,cmap='gray'),plt.title(' Gauss edges')
plt.xticks([]), plt.yticks([])

#plt.subplot(122),plt.imshow(e,cmap='gray'),plt.title(' Gausian')


plt.show()

