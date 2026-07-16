import cv2
import numpy as np

image = cv2.imread("/Users/bhavan/Documents/codes/Computer_Vision/Image-Editor/Input_Image/Input.jpg")

print(image.shape)
print(image.dtype)
print("original shape is : ", image.shape) # gives heigh, width, channels


while(True):

    cv2.imshow("Image", image)


    key = cv2.waitKey(1)

    if(key == ord('q')):
        print(ord('q'))
        break
    elif(key == ord('r')):
        width = int(input("Enter the desired width : "))
        height = int(input("enter the desired height : "))
        image = cv2.resize(image,(width, height)) # whenever we resize we pass width, height but in print it gives height first then width
        print("resized image shape is : ", image.shape)
        
    



cv2.destroyAllWindows()

