import tesseract
import cv2
import cv2.cv as cv
import numpy as np
import pyttsx
engine = pyttsx.init()
engine.setProperty('rate', 100)

scale = 1
delta = 0
ddepth = cv2.CV_16S

gray=cv2.imread("Untitled.png")
cv2.namedWindow("Main")
cv2.imshow("Main", gray)
### trim the edges
cut_offset=23
gray=gray[cut_offset:-cut_offset,cut_offset:-cut_offset]

### convert to gray color
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

### edge enhancing by Sobeling
# Gradient-X
grad_x = cv2.Sobel(gray,ddepth,1,0,ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
#grad_x = cv2.Scharr(gray,ddepth,1,0)

# Gradient-Y
grad_y = cv2.Sobel(gray,ddepth,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
#grad_y = cv2.Scharr(gray,ddepth,0,1)

abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
abs_grad_y = cv2.convertScaleAbs(grad_y)
gray = cv2.addWeighted(abs_grad_x,0.4,abs_grad_y,0.4,0)

### Bluring
image1 = cv2.medianBlur(gray,5) 
image1[image1 < 50]= 255
image1 = cv2.GaussianBlur(image1,(31,13),0)     
color_offset=220
image1[image1 >= color_offset]= 255  
image1[image1 < color_offset ] = 0      #black

#### Insert White Border
offset=30
height,width = image1.shape
image1=cv2.copyMakeBorder(image1,offset,offset,offset,offset,cv2.BORDER_CONSTANT,value=(255,255,255)) 
cv2.namedWindow("Test")
cv2.imshow("Test", image1)
cv2.imwrite("an91cut_decoded.jpg",image1)

### tesseract OCR
api = tesseract.TessBaseAPI()
api.Init(".","eng",tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)
#as suggested by zdenko podobny <zdenop@gmail.com>, 
#using PSM_SINGLE_BLOCK will be more reliable for ocr-ing a line of word. 
#api.SetPageSegMode(tesseract.PSM_SINGLE_BLOCK)
height1,width1 = image1.shape
channel1=1
image = cv.CreateImageHeader((width1,height1), cv.IPL_DEPTH_8U, channel1)
cv.SetData(image, image1.tostring(),image1.dtype.itemsize * channel1 * (width1))
tesseract.SetCvImage(image,api)
text=api.GetUTF8Text()
conf=api.MeanTextConf()
image=None
print "..............."
print "Ocred Text: %s"%text
engine.say(text)
engine.runAndWait()
print "Cofidence Level: %d %%"%conf
cv2.waitKey(0)
cv2.destroyWindow("Test")
cv2.destroyWindow("Main")
api.End()
