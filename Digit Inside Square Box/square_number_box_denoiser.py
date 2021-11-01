#my own soolution for my own stackoverflow post : https://stackoverflow.com/questions/69736363/ocr-how-to-recognize-numbers-inside-square-boxes-using-python/69798788#69798788

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
  

def square_number_box_denoiser(image_path="/content/9.png",is_resize = False, resize_width = 768):
    '''
    ref : https://pretagteam.com/question/removing-horizontal-lines-in-image-opencv-python-matplotlib

    Args : 
      image_path (str) : path of the image containing numbers/digits inside square box
      is_resize (int) : whether to resize the input image or not? default : False
      resize_width (int) : resizable image width for resizing the image by maintaining aspect ratio. default : 768 

    '''
    img=cv2.imread(image_path)
    if(is_resize):
      print("resizing...")
      img = imutils.resize(img, width=resize_width)
    image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)

    # create figure
    fig = plt.figure(figsize=(20, 20))
    # setting values to rows and column variables
    rows = 3
    columns = 3

    fig.add_subplot(rows,  columns, 1)
    plt.imshow(img)
    fig.add_subplot(rows,  columns, 2)
    plt.imshow(thresh)
    fig.add_subplot(rows,  columns, 3)
    plt.imshow(detected_lines)
    fig.add_subplot(rows,  columns, 4)
    plt.imshow(image)
    fig.add_subplot(rows,  columns, 5)
    plt.imshow(result)
    result = cv2.rotate(result,cv2.ROTATE_90_COUNTERCLOCKWISE)
    fig.add_subplot(rows,  columns, 6)
    plt.imshow(result)
    cv2.imwrite("result.jpg", result)

    plt.show()

