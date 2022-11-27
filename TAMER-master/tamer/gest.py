import cv2  #opencv
import mediapipe as mp
import time
import numpy as np
import math
def get_gest(cap,mpHands,hands,mpDraw):

    length = 0
    length2 = 0
    # mesure le temps de calcul pour chaque 
    pTime = 0 #temps avant calcul
    cTime = 0 #temps apres calcul
    
    # positions des keypoints de gest
    lmList = []
    #indcie de frame
    indice = 0
    #on prend deux premiers frames
    while indice<2 :
        indice +=1
        # image de camera
        success, img = cap.read()
        
        # mettre img a forme cv2
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detecter avec mediapipe.solutions.hands
        results = hands.process(imgRGB)
    
        # verifier s'il y a plusieurs mains dans img
        if results.multi_hand_landmarks: 
            for handlms in results.multi_hand_landmarks:
                
                # obtenir les positions (entre 0 et 1)
                for index, lm in enumerate(handlms.landmark):
                    
                    
                    h, w, c = img.shape 
                    
                    # mettre les position de [0,1] a la taille d'image
                    cx ,cy =  int(lm.x * w), int(lm.y * h) 
                    
                    # sauvegarder les positions
                    lmList.append([index, cx, cy])
                        
                    # draw les keypoints sur l'image
                    cv2.circle(img, (cx,cy), 12, (0,0,255), cv2.FILLED)
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                x3, y3 = lmList[2][1], lmList[2][2]
                length = math.hypot(x2 - x1, y2 - y1)
                length2 = math.hypot(x3 - x1, y3 - y1)
                print(length,length2)
                # connecter les keypoints
                mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS) 
            
        # temps de calcul      
        cTime = time.time()      
        # calculer fps
        fps = 1/(cTime-pTime)
        # reset temps de calcul
        pTime = cTime
        
        # affichage de fps
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        
        # affichage d'image
        cv2.imshow('Image', img)  
        if cv2.waitKey(10) & 0xFF==27:  #chaque frame duree 10 ms
            break
    return length,length2


if __name__ == '__main__':
    cap = cv2.VideoCapture(0) 

    mpHands = mp.solutions.hands 
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    t1 = time.time()
    l1,l2 = get_gest(cap,mpHands,hands,mpDraw)
    t2 = time.time()
    print(t2-t1)
    print('l',l2)

    # release camera
    cap.release()
    cv2.destroyAllWindows()
    print('finish')