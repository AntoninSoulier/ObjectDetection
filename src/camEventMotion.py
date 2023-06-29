import cv2
import datetime

# Drop Keys
# CYTech\PFE\Data\CamEventDropKey.mp4

# Drop Tennis ball
# Data\\tennis_camEvent_vdo.mp4

cap = cv2.VideoCapture("Data\\tennis_camEvent_vdo.mp4")  

ret, frame1 = cap.read()
ret, frame2 = cap.read()

start = datetime.datetime.now()
timer_lst = []

while(cap.isOpened()):

    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if(cv2.contourArea(contour) < 500 or cv2.contourArea(contour) > 800): 
            continue

        end = datetime.datetime.now()
        t = end-start
        timer_lst.append(t.microseconds/1000)

        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (300,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
    
    if(timer_lst):
        cv2.putText(frame1, str("Time of detection: {}".format(timer_lst[0])) + " ms", (-235,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

    for i in range(5000):
        cv2.imshow('feed', frame1)
    frame1 = frame2
    ret , frame2 = cap.read()

    if(cv2.waitKey(40) == 27):
        break

cv2.destroyAllWindows()
cap.release()

print(timer_lst[0])
