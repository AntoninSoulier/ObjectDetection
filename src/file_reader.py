import dv_processing as dv
import cv2 as cv

# Open the file
reader = dv.io.MonoCameraRecording("CYTech/PFE/malo-gesture.aedat4")

# Get and print the camera name
print(f"Opened an AEDAT4 file which contains data from [{reader.getCameraName()}] camera")

'''
while(reader.isRunning()):
    events = reader.getNextEventBatch()
    if(events is not None):
        print(f'{events}')
'''

cv.namedWindow("Preview", cv.WINDOW_NORMAL)

lastTimeStamp = None

while reader.isRunning():
    # Read a frame from the camera
    frame = reader.getNextFrame()

    if frame is not None:
        # Print the timestamp of the received frame
        print(f"Received a frame at time [{frame.timestamp}]")

        # Show a preview of the image
        cv.imshow("Preview", frame.image)

        # Calculate the delay between last and current frame, divide by 1000 to convert microseconds
        # to milliseconds
        delay = (2 if lastTimestamp is None else (frame.timestamp - lastTimestamp) / 1000)

        # Perform the sleep
        cv.waitKey(delay)

        # Store timestamp for the next frame
        lastTimestamp = frame.timestamp
