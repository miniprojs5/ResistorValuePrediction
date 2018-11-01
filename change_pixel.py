import numpy as np
    import cv2
     
    im = cv2.imread('/home/user/resistor/ResistorValuePrediction/resis5.jpg')
    im[np.where((im >[240,240,240]).all(axis = 2))] = [255,255,255]
    cv2.imwrite('output.png', im)
