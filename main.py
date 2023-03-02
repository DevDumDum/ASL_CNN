from asyncio.windows_events import NULL
import mediapipe as mp
import cv2
import numpy as np
import uuid
from pathlib import Path
import math
import matplotlib.pyplot as plt
import os
import time

from predict import predict_img

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
img_height = 100
img_width = 100
cwd = os.getcwd()
train_dir = cwd+"/archive/train/"
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def cal_dist(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def hand_position(image, results, frame_width, frame_height):
    output_image = image.copy()
    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_landmarks =  results.multi_hand_landmarks[hand_index]

        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

        index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

        middle_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

        ring_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
        ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y

        pinky_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
        pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y

        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y


        thumb_pos = cal_dist(wrist_x, wrist_y, thumb_tip_x, thumb_tip_y)
        index_pos = cal_dist(wrist_x, wrist_y, index_tip_x, index_tip_y)
        middle_pos = cal_dist(wrist_x, wrist_y, middle_tip_x, middle_tip_y)
        ring_pos = cal_dist(wrist_x, wrist_y, ring_tip_x, ring_tip_y)
        pinky_pos = cal_dist(wrist_x, wrist_y, pinky_tip_x, pinky_tip_y)

    return output_image ,results

def draw_box(image, results, s_img, temp, letter, padd_amount = 20):
    output_image = image.copy()

    height, width, _ = image.shape

    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        x_coordinates = np.array(landmarks)[:,0]
        y_coordinates = np.array(landmarks)[:,1]

        # Get the bounding box coordinates for the hand with the specified padding.
        x1  = int(np.min(x_coordinates) - padd_amount) #left
        y1  = int(np.min(y_coordinates) - padd_amount)      #top
        x2  = int(np.max(x_coordinates) + padd_amount) #right
        y2  = int(np.max(y_coordinates) + padd_amount)      #bottom

        y_distance = y2 - y1
        x_distance = x2 - x1
        
        if x_distance > y_distance:
            box_w = (x_distance - y_distance)/2
            x1  -= int(padd_amount)              #left
            y1  -= int(padd_amount + box_w)      #top
            x2  += int(padd_amount)              #right
            y2  += int(padd_amount + box_w)      #bottom
        else:
            box_w = (y_distance - x_distance)/2
            x1  -= int(padd_amount + box_w)      #left
            y1  -= int(padd_amount)              #top
            x2  += int(padd_amount + box_w)      #right
            y2  += int(padd_amount)              #bottom


        box = [x1,x2,y1,y2]
        label = "Letter: "+letter
        # Draw the bounding box around the hand on the output image.
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (55,255, 255), 2, cv2.LINE_8)

        # Write the classified label of the hand below the bounding box drawn.
        cv2.putText(output_image, label, (x2-100, y2+25), cv2.FONT_HERSHEY_COMPLEX, .5, (20,255,155), 1, cv2.LINE_AA)
    return output_image, temp, box

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 2) as hands:
    text = ""
    temp = {}
    status = 0
    cropped_status = 0
    pre_status = 0
    letter = ""
    confidence = 0
    temp_img = None
    
    while cap.isOpened():
        cropped = None
        ret, frame = cap.read()
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        s_img = image.copy()
        s_img = cv2.cvtColor(s_img, cv2.COLOR_RGB2BGR)
        #detections
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #render results
        if results.multi_hand_landmarks:
            image, results = hand_position(image, results, frame_width, frame_height)
            image, temp, box = draw_box(image, results, s_img, temp, letter)

            if (box[1] < frame_width-10  and box[0] > 10 and box[2] > 10 and box[3] < frame_height-10) and pre_status == 1:
                #cv2.imwrite('s.jpg',cropped) #save hand
                cropped = s_img[box[2]:box[3],box[0]:box[1]] #without skelly
                #cropped = image[box[2]:box[3],box[0]:box[1]] #with skelly
                cropped_status = 1
                status = 1
                pre_status = 0
                temp_img = None

        if status == 1:
            if (cropped_status == 1):
                letter, confidence, img, temp_img = predict_img(cropped)
                cropped_status = 0
                temp = cropped
            else:
                if temp_img is None:
                    cropped = temp
                    img = s_img[box[2]:box[3],box[0]:box[1]]
                    img = cv2.resize(img, (img_height,img_width), interpolation= cv2.INTER_LINEAR)
                    temp_img = img
                else:
                    temp_img = cv2.resize(temp_img, (img_height,img_width), interpolation= cv2.INTER_LINEAR)
                    img = temp_img

            if (letter != None):
                image[0:80, 0:100] = [0,0,0]
                image[frame_height - 50:frame_height, 0:frame_width] = [0,0,0]
                loc = train_dir+letter+"/"+letter+"_1.jpg"
                #print(str(loc))
                try:
                    img2 = cv2.imread(loc)
                    img = cv2.resize(img, (img_height,img_width), interpolation= cv2.INTER_LINEAR)
                    img2 = cv2.resize(img2, (img_height,img_width), interpolation= cv2.INTER_LINEAR)
                    image[80:80+img_height, 0:img_width] = img2
                    image[frame_height-img_height:frame_height, frame_width-img_width:frame_width] = img
                except Exception as e:
                    pass

                image = cv2.putText(image, "P: "+letter , (10,30), cv2.FONT_HERSHEY_SIMPLEX, .5, (20,255,155), 1)
                image = cv2.putText(image, "C: {:.2f}".format(confidence) , (10,60), cv2.FONT_HERSHEY_SIMPLEX, .5, (20,255,155), 1)
                
                #subtitle
                image = cv2.putText(image, "> "+text , (30,frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, .8, (155,255,150), 2)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q') or k == 27:
            break
        if  k == 32:
            text += letter
        if k == 8:
            text = text[:-1]
        pre_status = 1
        cv2.imshow('ASL Alphabet Recognition', image)


cap.release()
cv2.destroyAllWindows()