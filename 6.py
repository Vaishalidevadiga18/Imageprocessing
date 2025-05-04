import numpy as np
import cv2
import matplotlib.pyplot as plt
 
image=cv2.imread(r'C:\Users\VAISHALI\Downloads\CV\download (1).jpg')
if image is None:
    raise FileNotFoundError("the image 'ip1.jpg' was not found.")
gaussian_blur=cv2.GaussianBlur(image,(5,5),0)
median_blur=cv2.medianBlur(image,5)
laplacian=cv2.Laplacian(image,cv2.CV_64F)
laplacian=cv2.convertScaleAbs(laplacian)


titles=['Original Image','Gaussain Blur','Median Blur','Laplacian Filter']
images= [image,gaussian_blur,median_blur,laplacian]


plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
