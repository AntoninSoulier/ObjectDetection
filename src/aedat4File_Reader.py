import numpy as np
import cv2
import aedat

events = aedat.Decoder("Data\malo-gesture.aedat4")
frame = np.zeros((128,128))
num_events_processed = 0

for event in events:
    t = list(event.values())[1].tolist()
    l = list(t[0])
    x, y, polarity = l[1], l[-2], l[-1]
    if(polarity == True):
        frame[y,x] += 1

    num_events_processed += 1
    
    #Display each frame after processing a fixed number of events (every 1000 events).
    if num_events_processed % 1000 == 0:
        frame = cv2.GaussianBlur(frame,(5,5), 0)

        _, treshhold = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

        treshhold = cv2.convertScaleAbs(treshhold)

        contours, _ = cv2.findContours(treshhold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        for i in range(10000):
            cv2.imshow("Movement", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # exit on pressing 'q' key
            break

cv2.destroyAllWindows()
