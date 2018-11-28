
#from PIL import Image, ImageColor
import numpy as np
import cv2
import matplotlib as plt
from math import sqrt

color_name_list=['black','brown','red','orange','yellow','green','blue','violet','grey','white','gold','noncolr','noncolor','noncolor','noncolor','noncolor','noncolor6',
'noncolr7','green','blue','noncolor','noncolor','noncolor']
color_value_list=[[0,0,0],[59,35,11],[255,0,0,],[255,165,0],[255,255,0],[0,255,0],[0,0,255],[148,0,211],[190,190,190],[255,255,255],[255,215,0],[165,148,96],[252,249,216],[173,168,110],[197,196,152],[174,157,101],[161,142,99],[191,187,139],[97,180,50],[15,18,23],[176,172,161],[164,154,144]]

pixel_val = []

color_ls = []

distance_matrix =['4000000']
import cv2 
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

#loading test image
img= cv2.imread('/home/gayathri/resistor/ResistorValuePrediction/enhance1.jpg')
img = cv2.resize(img,(360,480))


rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
rgb1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)



#find threshold values 
high_thres,thres_img=cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
low_thres=0.5*high_thres

#Smoothening and noise reduction using Bilateral filter

blur2 = cv2.bilateralFilter(gray,2,high_thres,low_thres)

#Edge Detection : Canny Edge detection Method

edges3 = cv2.Canny(blur2,300,350)
#print(edges)


#plt.subplot(121),plt.imshow(edges3,cmap='gray'),plt.title(' Bi  2')
#plt.xticks([]), plt.yticks([])

plt.show()

#finding contours 
_,contours,_=cv2.findContours(edges3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
no_of_contours=len(contours)
draw=cv2.drawContours(rgb,contours,-1,(0,0,255),2)

#==================== cropping the image =====================

height,width,_=draw.shape
print(draw.shape)
start_row,start_col=int(height*0.10),int(width*0.10)
end_row,end_col=int(height*0.95),int(width*0.95)

cropped=rgb1[start_row:end_row,start_col:end_col]


#print(edges3) 









#img = cv2.cvtColor(rgg, cv2.COLOR_BGR2HSV)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = edges3

def color_num(x):
   for i in range(len(color_name_list)):
       if color_name_list[i]==x:
          return i


def resistance(c1):
   
     
    #print(idx)
    cn1=color_num(c1[0])
    cn2=color_num(c1[1])
    cn3=color_num(c1[2])
    #print(cn3,'cn3')
    sumx=cn1*10+cn2
    powx=pow(10, int(cn3))
    return sumx*powx


def find_all(ls):
    flg=1
    for i in range(len(ls)):
        if ls[i]>0:    
            return 1
def find_distance(a,b):
    return sqrt(sum( (a - b)**2 for a, b in zip(a, b)))
	

def find_val_list(pixel):
    for i in range(len(color_value_list)):
        distance=find_distance(color_value_list[i],pixel)
	distance_matrix.append(distance)
    ind=distance_matrix.index(min(distance_matrix))
    if ind>19:
        ind=19
#print(ind)
    distance_matrix[:] = []
    return color_name_list[ind] 

def find_color():
    for i in range(len(pixel_val)):
        pixel=np.fliplr([pixel_val[i]])[0]
	#print(pixel)
        color_pix=find_val_list(pixel)
	color_ls.append(color_pix)
    return color_ls


def ROUND(a):
    return int(a + 0.5)

def drawDDA(x1,y1,x2,y2,img):
    x,y = x1,y1
    length = (x2-x1) if (x2-x1) > (y2-y1) else (y2-y1)
    dx = (x2-x1)/float(length)
    dy = (y2-y1)/float(length)
#    print 'x = %s, y = %s' % (((ROUND(x),ROUND(y))))
    for i in range(length):
        x += dx
	y += dy
	#print 'x = %s, y = %s' % (((ROUND(x),ROUND(y))))
	
	try:
	   pixel_val.append(img[ROUND(x),ROUND(y)])
           #cv2.circle(img,(int(ROUND(y)),int(ROUND(x))),2,(0,0,255),1)
	except IndexError:
		pass
	continue
			
	
#finding statical crop value

for i in range(gray.shape[0]):
    flag=0
    flag=find_all(gray[i])
    #print(flag)
    if  flag==1:
        x=i
        y=0
#print(x,'y=')
    #print()   
print('2nd')
        #print(gray[i,j])   
for i in reversed(xrange(gray.shape[0])):
    flag=0
    flag=find_all(gray[i])
    if flag==1:
        x1=i
        y1=gray.shape[1]




ptx=(x+x1)/2
pty=0
ptxmax=(x+x1)/2
ptymax=y1





#drwing line and making its color finding



print((img.shape))
#cv2.line(img,(0,36),(92,36),(255,0,0),5)
#print("dda")
pix = drawDDA(ptx,pty,ptxmax,ptymax,img)
#print(color_value_list[0])
#print(pixel_val[0])
cols = find_color()
#print(cols)
#print (cols)
import re
pattern = re.compile("non")
newx=[]
newx2=[]
for i in range(len(cols)-1):
    if pattern.match(cols[i]):
        continue
    else:
        if cols[i]==cols[i+1]:		
	    newx.append(cols[i])
	else:
	    newx.append('no')
#print(newx)

for i in range(len(newx)):
    if newx[i]=='no':
        if newx[i-1]!='no':
            newx2.append(newx[i-1])


idx=0
for i in range(len(newx2)):
    if newx2[i]!='white':
        idx=i
        break


value=resistance(newx2[idx:])
print(newx2[idx:])
print(value,'ohms')








cv2.line(img,(pty,ptx),(ptymax,ptxmax),(255,0,0),5)
cv2.rectangle(img,(y,x),(y1,x1),(0,0,255),3)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
