# PHOENIX

SOCIAL DISTANCING AND FACE MASK DETECTOR

USING COMPUTER VISION AND DEEP LEARNING TO PREDICT FACE MASKS WEARING/NOT AND SOCIAL DISTANCING ABIDING/VIOLATING

BASED ON TRAINED MODEL TO DETECT FACE MASKS AND SOCIAL DISTANCING USING YOLOV3 MODEL

# Download yolov3 weights from the link : https://pjreddie.com/media/files/yolov3.weights

# Approach has 3 parts :
1.Social distancing measurement. 

2.Face Mask Recognition

3.Face Mask properly worn or not.

A GUI is built and multithreading approach is taken to render faster solution at realtime. 
Social distancing measurement using YOLOv3 model for person detection and distance calculation by euclidean method.
Face mask detection using face detection(caffe model) and mask detection model trained by medicated mask dataset. 
Face Mask properly worn or not using facial landmarks .

# Solution can be implemented to achieve the benefits:

This system works at realtime providing signal outputs of people who are abiding by the rules and who are not, thereby continuously tracking and returning indexes where rules fail. 

Also video is stored in realtime to be monitored.

# Detailed explanation on the Future State of your solution.

REALTIME APPLICATION CAN BE USED IN PUBLIC PLACES SUCH AS OFFICES,MARKETS,AIRPORTS ETC. WHERE SOCIAL DISTANCING AND PROPERLY MASK WEARING IS MANDATORY.
Fast in deployment with any external source such as CCTV or IOT
Renders solutions that are inevitable during this serious pandemic situation 

