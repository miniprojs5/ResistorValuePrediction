import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('/home/user/resistor/ResistorValuePrediction/Resistor images/vertical/20181031_151241.jpg')
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
num_rows, num_cols = img1.shape
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
img_rotation = cv2.warpAffine(rgb, rotation_matrix, (num_cols, num_rows))
plt.subplot(121),plt.imshow(img_rotation),plt.title('cropped')
plt.xticks([]), plt.yticks([])
plt.show()
img_rotation=cv2.cvtColor(img_rotation,cv2.COLOR_RGB2BGR)
#plt.savefig('/home/user/resistor/ResistorValuePrediction/Resistor images/processed/20181031_151241.jpg',img_rotation)
cv2.imwrite('/home/user/resistor/ResistorValuePrediction/Resistor images/processed/res1_1000_5.jpg',img_rotation)

