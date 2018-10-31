import cv2 
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
#np.set_printoptions(threshold=np.inf)


img= cv2.imread('/home/user/resistor/ResistorValuePrediction/resis5.jpg')

#hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#bilateralFilter(image, dstBila, kernel_length, kernel_length*2, kernel_length/2);

rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rgb1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


#edges = cv2.Canny(blur,100,200)

#find threshold values 
high_thres,thres_img=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
low_thres=0.5*high_thres

#Smoothening and noise reduction using Bilateral filter
blur = cv2.bilateralFilter(rgb,5,high_thres,low_thres)
blur1 = cv2.bilateralFilter(rgb,3,high_thres,low_thres)
blur2 = cv2.bilateralFilter(gray,2,high_thres,low_thres)

#Trying smoothening with Gaussian filetr
gaussian_blur=cv2.GaussianBlur(gray,(5,5),0)

#Edge Detection : Canny Edge detection Method
edges1 = cv2.Canny(blur,100,200)
edges = cv2.Canny(gaussian_blur,100,200)
edges2 = cv2.Canny(blur1,100,200)
edges3 = cv2.Canny(blur2,100,200)

#===========Showing original and Blurred image=============
plt.subplot(121),plt.imshow(rgb),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('blurred')
plt.xticks([]), plt.yticks([])
plt.show()
#=====================Plotting the edges=====================================

plt.subplot(121),plt.imshow(edges,cmap='gray'),plt.title(' Gauss edges')
plt.xticks([]), plt.yticks([])
plt.show()
plt.subplot(121),plt.imshow(edges1,cmap='gray'),plt.title(' Bi lateral Edges 5')
plt.xticks([]), plt.yticks([])


#plt.subplot(122),plt.imshow(e,cmap='gray'),plt.title(' Gausian')



plt.subplot(122),plt.imshow(edges2,cmap='gray'),plt.title(' Bilateral  edges 3')
plt.xticks([]), plt.yticks([])

plt.show()
plt.subplot(121),plt.imshow(edges3,cmap='gray'),plt.title(' Bi  2')
plt.xticks([]), plt.yticks([])

plt.show()

#finding contours 
_,contours,_=cv2.findContours(edges3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
no_of_contours=len(contours)
#print(contours)
#cont=np.array(contours)

draw=cv2.drawContours(rgb,contours,-1,(0,0,255),2)

#plotting contours
plt.subplot(121),plt.imshow(draw,cmap='gray'),plt.title('drawcontours')
plt.xticks([]), plt.yticks([])
plt.show()

#==================== cropping the image =====================

height,width,_=draw.shape
print(draw.shape)
start_row,start_col=int(height*0.2),int(width*0.25)
end_row,end_col=int(height*0.5),int(width*0.75)

cropped=rgb1[start_row:end_row,start_col:end_col]

cropped1=rgb1[start_row:end_row,start_col:end_col]
plt.imsave('cropped.jpg',cropped)

plt.subplot(121),plt.imshow(cropped1),plt.title('cropped')
plt.xticks([]), plt.yticks([])
plt.show()

#===================Band Extraction========================

cropped_gray=cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)

#find threshold values 
high_thres,thres_img=cv2.threshold(cropped_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
low_thres=0.5*high_thres

cropped_blur = cv2.bilateralFilter(cropped_gray,3                                                                                                                                                                                         ,high_thres,low_thres)
cropped_edges = cv2.Canny(cropped_blur,100,200)

#plotting cropped edges

plt.subplot(122),plt.imshow(cropped_edges,cmap='gray'),plt.title('cropped')
plt.xticks([]), plt.yticks([])
plt.show()

#finding contours in cropped images

_,contours,_=cv2.findContours(cropped_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
no_of_contours=len(contours)
#print(contours)
print(no_of_contours)

#Seeing contours 
#area_list=[]
#for cnt in contours:
   # rect=cv2.minAreaRect(cnt)
    #box=cv2.boxPoints(rect)
    #box=np.int0(box)
    
 #   draw1=cv2.drawContours(cropped1,[cnt],0,(0,0,255),2)
    #draw1=cv2.cvtColor(draw,cv2.COLOR_RGB2BGR)
  #  area_list.append(cv2.contourArea(cnt))
   # area=cv2.contourArea(cnt)
   # print("Area of contour number  ",area)
    #cv2.imshow('image',cropped1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
#====================printing contour with larger area ==========
#print(sorted(area_list))



#finding dominant colors



class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        img = cv2.imread(self.IMAGE)
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

    
img = '/home/user/resistor/ResistorValuePrediction/cropped.jpg'
#img_cv=cv2.imread(cropped)
clusters = 6
dc = DominantColors(img,clusters) 
colors = dc.dominantColors()
print(colors)

#scanning parts of images using window frames

