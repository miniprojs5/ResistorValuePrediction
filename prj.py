import cv2
 
img = cv2.imread('/home/user/resistor/res.jpeg', cv2.IMREAD_UNCHANGED)
 
print('Original Dimensions : ',img.shape)
 
width = 200
height = 100
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
#========================================================
#accessed all files in a folder AND RESIZED TO 200 100

import cv2
import os
import glob
img_dir = "/home/user/resistor/test/*.jpg"  
data_path = os.path.abspath(img_dir)
files = glob.glob(data_path)
print(files)
width=250
height=100
for f in files:
    img=cv2.imread(f,cv2.IMREAD_UNCHANGED)
    dim=(width,height)
    resized=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    print("resized dimensions",resized.shape)
    
