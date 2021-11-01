
!pip install deepface
from deepface import DeepFace
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def crop_face_from_doc_img(img_path = '/content/v.jpg',backbone = 3):
    '''
    assume the orientation of input image is unknown
    References :
    1. https://github.com/serengil/deepface
    2. https://viso.ai/computer-vision/deepface/

    Args : 
      img_path (str) : path of the document/nid image containing face
      backbone (int) : choice of algorithms -> 0 : 'opencv', 1 : 'ssd', 2 : 'dlib', 3 : 'mtcnn', 4 : 'retinaface'
    '''
    im = Image.open(img_path)
    for i in range(1, 360, 20):
      try: 
        #rotation angle in degree
        #rotated = ndimage.rotate(cv2.imread('/content/MicrosoftTeams-image (5).png'), i)
        rotated = im.rotate(i)
        rotated = np.asarray(rotated)
        
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']
        
        img = DeepFace.detectFace(rotated, detector_backend = backends[backbone]) #, enforce_detection=False
        # faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(rotated)
        # if(faces.shape == (1, 4)):
        #     print("face detected............... ")
        #     for (x, y, w, h) in faces:
        #       cv2.rectangle(rotated, (x,y), (x+w, y+h), (255,0,0), 2)   
        #     plt.imshow(rotated)
        #     break
        print("Used = ",backends[backbone])
        plt.imshow(img)
        name = img_path.rstrip(os.sep) # strip the slash from the right side
        name = os.path.basename(name)
        frame_normed = 255 * (img - img.min()) / (img.max() - img.min())
        img = np.array(frame_normed, np.int)
        cv2.imwrite(f"cropped_{name}", img)
        break
        
      except:
        #pass
        print(f"{i} degree rotation not working")

crop_face_from_doc_img(img_path = "/content/t1.png",backbone = 2)
