import cv2
import numpy as np
import os
import pyautogui


def start():
    #print("123")

    output = "OUTPUT/video.avi"
    img = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # get info from img
    height, width, channels = img.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))



    while True:
        # make a screenshot
        img = pyautogui.screenshot()
        # convert these pixels to a proper numpy array to work with OpenCV
        frame = np.array(img)
        # convert colors from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # write the frame
        out.write(frame)
        # show the frame
        #cv2.imshow("screenshot", frame)
        # if the user clicks q, it exits
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    # make sure everything is closed when exited
    cv2.destroyAllWindows()
    out.release()

if __name__=="__main__":
    start()