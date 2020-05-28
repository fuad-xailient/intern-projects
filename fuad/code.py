import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    
img = cv2.imread('DATA/Nadia_Murad.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
display_image(img,cmap=None)

haar_cascade_face = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')
faces_rects = haar_cascade_face.detectMultiScale(img,scaleFactor=1.2,minNeighbors=5)

print('Faces found: ',len(faces_rects),"Photo 1")

for (x,y,w,h) in faces_rects:
    cv2.rectangle(img,(x,y),(x+w,y+h),color=(255,0,0),thickness=4)
    
display_image(img)


group_img = cv2.imread('DATA/two_girls.jpg')
group_img = cv2.cvtColor(group_img,cv2.COLOR_BGR2RGB)
display_image(group_img,cmap=None)

def detect_faces(cascade,new_img,scaleFactor=1.2):
    faces_rect = haar_cascade_face.detectMultiScale(new_img,scaleFactor=scaleFactor,minNeighbors=5)
    for (x,y,w,h) in faces_rect:
        cv2.rectangle(new_img,(x,y),(x+w,y+h),color=(0,0,255),thickness=9)

faces = detect_faces(haar_cascade_face,group_img)
display_image(group_img)

faces_rects = haar_cascade_face.detectMultiScale(group_img,scaleFactor=1.5,minNeighbors=5)
print('Faces found: ',len(faces_rects),'Photo 2')