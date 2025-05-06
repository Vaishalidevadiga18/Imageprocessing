import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

image=cv2.imread(r"C:\Users\VAISHALI\Downloads\CV\download (1).jpg")
if image is None:
    raise FileNotFoundError("The path is not found")
_,binary_image=cv2.threshold(image,127,255,cv2.THRESH_BINARY)
kernel=np.ones((3,3),np.uint8)
eroded=cv2.erode(binary_image,kernel,iterations=1)
dilated=cv2.dilate(binary_image,kernel,iterations=1)
opening=cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE,kernel)
titles=['Original Image',' Erosion',"Dilation","Opening",'Closing']
images=[binary_image,eroded,dilated,opening,closing]
plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()