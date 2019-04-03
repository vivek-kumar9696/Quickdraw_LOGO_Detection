
# coding: utf-8

# In[ ]:


import cv2
from keras.models import load_model
import numpy as np
from collections import deque


from keras.preprocessing import image
import keras
import os


# In[ ]:


model1 = load_model('mob_logo_model.h5')
val = ['Adidas','Apple','BMW','Citroen','Fedex','HP','Mcdonalds','Nike','none','Pepsi','Puma']
pred_class  = 8


# In[ ]:


def nothing(x):
    pass
 
cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")
 
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


# In[ ]:



def main():
    logos = get_logos()
    cap = cv2.VideoCapture(0)
    Lower_green = np.array([10,130,130])
    Upper_green = np.array([40,255,255])
    pts = deque(maxlen=512)
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    value = np.zeros((224,224,3), dtype = np.uint8)
    #print(blackboard)
    
    digit = np.zeros((200, 200, 3), dtype=np.uint8)
    pred_class = 8

    while (cap.isOpened()):
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.rectangle(img,(400,250),(624,474),(255,0,255),5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        
        Lower_green= np.array([l_h, l_s, l_v]) # use the trackbars to customize the colour to track to make the doodles
        Upper_green = np.array([u_v, u_s, u_v]) #0,131,157     179,255,255 (orange color settings)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.inRange(hsv, Lower_green, Upper_green)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center = None
        

        if len(cnts) >= 1:
            cnt = max(cnts, key=cv2.contourArea)
    
            
            
            #print(cnt)
            if cv2.contourArea(cnt) > 200:
                
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                pts.appendleft(center)
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
                    cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
                    
        elif len(cnts) == 0:
            
            if len(pts) != []:
                
                blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
        
                
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                if len(blackboard_cnts) >= 1:
                    cnt = max(blackboard_cnts, key=cv2.contourArea)
                    #print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        value = blackboard[250:474, 400:624]
                       
                        pred_probab, pred_class = keras_predict(model1, value)
                        print(val[pred_class], pred_probab)

            pts = deque(maxlen=512)
            
            
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
          
            
            img = overlay(img, logos[pred_class])
        cv2.imshow("Frame", img)
        cv2.imshow("Res", res)
        cv2.imshow("mask", mask)
        
        
        k = cv2.waitKey(10)
        if k == 27:
            break


# In[ ]:


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model1.predict(processed)[0]
    
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


# In[ ]:


def keras_process_image(img):
    img_array = image.img_to_array(img)
    
    img_array_expanded_dims = np.expand_dims(img_array, axis = 0)
    
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# In[ ]:


def get_logos():
    logos_folder = "../logo/"
    logos = []
    
    
    
    for logo in range(len(os.listdir(logos_folder))):
        
        
        logos.append(cv2.imread(logos_folder + str(logo) + '.png', cv2.IMREAD_UNCHANGED))
        
        print(logos)
        
    return logos


# In[ ]:


def overlay(image, logo):
    x,y,z = logo.shape
  
    #try:
    image[0:x, 0:y] = blend_transparent(image[0:x, 0:y ], logo)
    #except:
        #pass
    return image


# In[ ]:


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the BRG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


# In[ ]:


keras_predict(model1, np.zeros((224, 224, 3), dtype=np.uint8))
main()

