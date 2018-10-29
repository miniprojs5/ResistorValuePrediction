#finding dominant color in an image


import cv2
from matplotlib import pyplot as plt
import numpy as np


image= cv2.imread('/home/gayathri/resistor/ResistorValuePrediction/330_resistor.jpg')

# reshape the image to be a list of pixels
#image = image.reshape((image.shape[0] * image.shape[1], 3))

#covert to rgb
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

z=rgb.reshape((-1,3))
z=np.float32(z)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

k=6
ret,lablel1,center1=cv2.kmeans(z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center1=np.uint8(center1)
res1=center1[label1.flatten()]
output1=res1.reshape((img.shape))

k=5
ret,lablel1,center1=cv2.kmeans(z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center1=np.uint8(center1)
res1=center1[label1.flatten()]
output2=res1.reshape((img.shape))

k=4
ret,lablel1,center1=cv2.kmeans(z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center1=np.uint8(center1)
res1=center1[label1.flatten()]
output3=res1.reshape((img.shape))

output=[image,output1,output2,output3]
titles=['original','k=6','k=5','k=4']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(output[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
    plt.show()
	

