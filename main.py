import numpy as np
import cv2

# Capture video from file
cap = cv2.VideoCapture('video.mp4')

min_width_rec = 80
min_higth_rec = 80

line_position = 500

# Substructor
alg = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handler(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

detect = []

offset = 6

#counter = 0

while True:

    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)

    img_sub = alg.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(25,line_position), (1200,line_position),(255,127,0),3)
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rec) and (h>= min_higth_rec)
        if not validate_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, "Vehicle", (x, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,244,0),2)

        center = center_handler(x,y,w,h)
        detect.append(center)
        cv2.circle(frame,center,4,(0,0,255),-1)

    # Count doesn't work good
    """
    for (x,y) in detect:
        if y < (line_position+offset) and y > (line_position-offset):
            counter += 1
        cv2.line(frame,(25,line_position),(1200,line_position),(0,127,255),3)
        detect.remove((x,y))

        print("Vehicle Counter: " + str(counter))

    cv2.putText(frame, "Vehicle Counter: " + str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
"""

    if ret == True:

        Frame = cv2.resize(frame, (960,480))
        cv2.imshow('Vehicle Detection',Frame)
        #print(Frame)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()