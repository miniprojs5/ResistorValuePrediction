from PIL import ImageEnhance
from PIL import Image
import cv2

img = Image.open('/home/gayathri/resistor/ResistorValuePrediction/img.jpg')
#size=width,height=img.size
img = img.resize((200,200), Image.ANTIALIAS)
enhancer=ImageEnhance.Brightness(img)
enhancer=ImageEnhance.Contrast(img)
img=enhancer.enhance(1.5)
img.show(img)
img.save('/home/gayathri/resistor/ResistorValuePrediction/enhance.jpg')
