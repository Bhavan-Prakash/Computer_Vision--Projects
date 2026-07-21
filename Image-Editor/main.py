import cv2
import numpy as np

image = cv2.imread("/Users/bhavan/Documents/codes/Computer_Vision/Image-Editor/Input_Image/Input.jpg")

print(image.shape)
print(image.dtype)
print("original shape is : ", image.shape) # gives heigh, width, channels (height == rows, width == columns)


while(True):

    cv2.imshow("Image", image)


    key = cv2.waitKey(1)

    if(key == ord('q')):
        print(ord('q'))
        break
    elif(key == ord('s')):
        width = int(input("Enter the desired width : "))
        height = int(input("enter the desired height : "))
        image = cv2.resize(image,(width, height)) # whenever we resize we pass width, height but in print it gives height first then width
        print("resized image shape is : ", image.shape)
    elif(key == ord('f')):
        image = cv2.flip(image,0)
    elif(key == ord("r")):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif(key == ord("c")):
        image = image[200:700,100:500]
    elif(key == ord("d")):
        # image = cv2.line(image, (500,500), (1000,700), (255,0,0), 40)
        image = cv2.rectangle(image, (500,500), (1000,700), (0,0,255), 20)
        # image = cv2.rectangle(image, (500,500), (1000,700), (0,0,255), -1)
        # image = cv2.circle(image, (500,500), 20,(0,0,255), 20 )
    elif(key == ord("t")):
        cv2.putText(image, "hello nan", (1100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 10)

    



cv2.destroyAllWindows()

