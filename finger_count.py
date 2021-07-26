import cv2
import mediapipe as mp

# initialise instances and parameter values
mpHand=mp.solutions.hands
hands=mpHand.Hands(max_num_hands=2,min_detection_confidence=0.8,min_tracking_confidence=0.5)
mpDraw=mp.solutions.drawing_utils

# initialise webcam
cam=cv2.VideoCapture(0)

tips=(8,12,16,20) # serial numbers of finger tip landmarks


with hands as hand:

    while True:

        _,frame=cam.read()
        im=cv2.flip(frame,1)
        h,w,_=im.shape

        # convert to RGB and get the landmarks
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        hand_lm=hand.process(rgb)
        im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

        if hand_lm.multi_hand_landmarks:
            count = [] # for storing finger count

            # iterating over the landmarks
            for lm in hand_lm.multi_hand_landmarks:

                # count for thumb
                if lm.landmark[4].x < lm.landmark[2].x:

                    # get thumb tip coordinates
                    x,y=int(lm.landmark[4].x*w),int(lm.landmark[4].y*h)
                    count.append(1)

                # count for fingers
                for i,pts in enumerate(lm.landmark):
                    x,y=int(pts.x*w),int(pts.y*h)

                    # for only the finger tip coordinates
                    if i in tips:
                        if lm.landmark[i].y<lm.landmark[i-2].y:

                            # store finger count
                            count.append(1)

                # display finger count (including thumb)
                cv2.putText(im,'finger count : {}'.format(len(count)),(5,25),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)

                # draw hand landmarks
                mpDraw.draw_landmarks(im,lm,mpHand.HAND_CONNECTIONS)
        else :
            cv2.putText(im,'finger count : 0',(5,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)

        cv2.imshow('webcam',im)

        if cv2.waitKey(1) & 0xff==ord('q'):
            break

cam.release()
cv2.destroyAllWindows()