#  Fast & Light Guide OpenCV Functions Guide

## Introduction
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides various functions for image and video processing tasks. This guide provides an overview of some commonly used functions in OpenCV.
```python
import rospy
from clover import srv
from std_srvs.srv import Trigger
from aruco_pose.msg import MarkerArray
import math
from clover import long_callback
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover.srv import SetLEDEffect
from cv2 import aruco
import cv2
import datetime
```
## Image Processing Functions
### imread
Image processing functions in OpenCV enable developers to perform various operations on images, including reading, displaying, resizing, filtering, and edge detection. These functions are fundamental for many computer vision applications, from simple image manipulation to complex object detection and recognition tasks.
The imread function is used to read an image from a file into a NumPy array. It supports various image formats such as JPEG, PNG, BMP, etc. You can specify flags to control the color mode in which the image is read.
imshow is a function used to display an image in a window. It takes the image array and a window name as input and opens a new window displaying the image.
- **Description:** Reads an image from a file.
- **Syntax:** `cv2.imread(filename, flags)`
- **Example:** `img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)`
```python
rospy.init_node('flight')
bridge = CvBridge()
```

### imshow
The imshow function in OpenCV is a fundamental tool for visualizing images during the development and debugging of computer vision applications. It allows developers to display images within graphical user interfaces (GUIs) or interactive environments, facilitating the inspection of image processing results and algorithm outputs. With imshow, developers can quickly view images loaded from files or generated during runtime. This function takes two essential parameters: the window name and the image to be displayed. Once invoked, imshow creates a window with the specified name and renders the image within it.
- **Description:** Displays an image in a window.
- **Syntax:** `cv2.imshow(window_name, image)`
- **Example:** `cv2.imshow('image', img)`
```python
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)
image_pub = rospy.Publisher('Name_Topic_debug', Image)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)
car_cascade = cv2.CascadeClassifier('h/home/clover/Desktop/haarcascade_car.xml')
```

### resize
The resize function in OpenCV enables developers to adjust the dimensions of images, allowing for scaling, cropping, or aspect ratio manipulation. Resizing images is a fundamental operation in image processing and computer vision applications, often used to prepare data for further analysis or visualization.
This function accepts several parameters, including the source image (src) and the desired dimensions (dsize) of the output image. Developers can specify the dimensions either as a tuple (width, height) or individually as fx and fy, representing scale factors along the horizontal and vertical axes, respectively. Additionally, developers can choose an interpolation method to determine how pixel values are computed in the resized image.
- **Description:** Resizes an image.
- **Syntax:** `cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])`
- **Example:** `resized_img = cv2.resize(img, (width, height))`
```python
def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x   2 + telem.y   2 + telem.z   2) < tolerance:
            break
        rospy.sleep(0.2)
```

## Video Processing Functions
Video processing functions in OpenCV enable developers to work with video streams, capture frames from video sources, apply transformations, and perform various operations on video data. These functions are essential for tasks such as video analysis, object tracking, and surveillance systems. The VideoCapture function initializes a video capture object, which can be used to capture video frames from cameras, video files, or other video sources. It accepts an index parameter indicating the device's ID (for cameras) or the path to the video file. The read function grabs and decodes the next video frame from the capture object. It returns a tuple consisting of a Boolean value indicating whether the frame was successfully read and the frame itself.
### VideoCapture
- **Description:** Captures video from a camera or file.
- **Syntax:** `cv2.VideoCapture(index)`
- **Example:** `cap = cv2.VideoCapture(0)`
```python
def markers_callback(msg):
    global mark
    for marker in msg.markers:
        if not marker.id in mark:
            mark.append(marker.id)
            set_effect(r=0, g=255, b=0)
```

### read
The read function in OpenCV is used to retrieve frames from a video stream or file. It grabs the next available frame from the video source, decodes it, and returns it as a two-element tuple. The first element of the tuple is a boolean value indicating whether the frame was successfully retrieved, and the second element is the actual frame data. In this example, the read function is used within a loop to continuously read frames from a video file (video.mp4). Each frame is displayed using imshow, and the loop continues until the end of the video file is reached or the user presses the 'q' key. Finally, the capture object is released, and all OpenCV windows are closed.

The read function is essential for video processing tasks, allowing developers to access individual frames for analysis, manipulation, or display in real-time applications. Whether working with live video streams or pre-recorded video files, the read function provides a straightforward mechanism for accessing video data in OpenCV.
- **Description:** Grabs, decodes, and returns the next video frame.
- **Syntax:** `cv2.VideoCapture.read([image])`
- **Example:** `ret, frame = cap.read()`
```python
@long_callback
def image_callback(data):
    global mark, count_violations, count_rights, coordinates_violations
    full_img = bridge.imgmsg_to_cv2(data, 'bgr8')
    height, width, _ = full_img.shape
    roi_size = 65
    roi_x = (width - roi_size) // 2
    roi_y = (height - roi_size) // 2
    roi = full_img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = aruco.detectMarkers(roi, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(corners)):
            for j in range(len(corners[i])):
                corners[i][j][:, 0] += roi_x
                corners[i][j][:, 1] += roi_y 

        full_img = aruco.drawDetectedMarkers(full_img, corners, ids, borderColor=(255, 0, 0))
        for i in range(len(ids)):
            #marker_id = ids[i][0]
            #text = f"{marker_id}: free"
            text = "free"
            text_x = int(corners[i][0][0][0])
            text_y = int(corners[i][0][0][1]) + 20 
            cv2.putText(full_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml') 
    gray_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7)

    for (x, y, w, h) in cars:
        count_violations = count_violations + 1
        cv2.rectangle(full_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(full_img, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    image_pub.publish(bridge.cv2_to_imgmsg(full_img, 'bgr8'))
```

### release
The release function in OpenCV is used to release the resources associated with a video capture object. It should be called when you're done working with the video capture object to free up system resources and ensure proper cleanup. In this example, the release function is called after processing all frames from the video file. This ensures that the resources used by the video capture object (cap) are properly released when they're no longer needed.

Failing to call release can lead to resource leaks and potential issues, especially if your program runs for an extended period or if you're working with multiple video capture objects simultaneously. Therefore, it's good practice to always release video capture objects when you're done using them.
- **Description:** Closes the video file or capturing device. 
- **Syntax:** `cv2.VideoCapture.release()`
- **Example:** `cap.release()`
```python
def main_fly():
    # x = [0, 0,   0.54, 0.54, 1.08,  1.08,  1.35, 1.62, 1.62, 0]
    # y = [0, 2.7, 2.7,   0,   0,     2.7,   2.7,  2.7,  0,    0]
    x = [0, 0, 0]
    y = [0, 2.7, 0]
    z = 1
    j = 0

    while j < len(x):
        set_effect(r=0, g=0, b=0)
        print(f"Flight to x = {x[j]} | y = {y[j]} | z = {z}")
        navigate_wait(x=x[j], y=y[j], z=z, frame_id='aruco_map', speed=0.4)
        j = j+1
        rospy.sleep(1)
```

## Image Filtering Functions
Image filtering functions in OpenCV are essential tools for modifying the appearance of images by applying various filters and transformations. These functions allow developers to enhance image quality, reduce noise, extract features, and detect edges, among other tasks.

GaussianBlur
The GaussianBlur function applies a Gaussian blur to an image, which is a widely used technique for reducing image noise and smoothing out details. It convolves the image with a Gaussian kernel to calculate the weighted average of neighboring pixels, effectively blurring the image.
The Canny function is used for edge detection in images. It applies a multi-stage algorithm to detect a wide range of edges with high accuracy and low error rates. The Canny edge detector is widely used in computer vision applications for tasks such as object detection, shape recognition, and image segmentation.
### GaussianBlur
- **Description:** Blurs an image using a Gaussian filter.
- **Syntax:** `cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])`
- **Example:** `blurred_img = cv2.GaussianBlur(img, (5, 5), 0)`
```python
def create_report():
    global mark, count_violations, count_rights, coordinates_violations
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    address = "Кондратьевский 46"
    total_parking_spots = 24

    report = f"""
    Aкт об обследовании парковки 

    Дата:  {current_time}

     Место проведения обследования: 
    Парковка ТКУИК
    Адрес: {address}

     Результаты обследования: 
    1.  Состояние парковки:  
       - Общее количество парковочных мест: {total_parking_spots}
       - Количество свободных парковочных мест: {len(mark)}
       - Количество занятых парковочных мест: {total_parking_spots - len(mark)}

    2.  Состояние парковочных мест: 
       - Координаты парковочных мест, где обнаружены нарушения: {coordinates_violations}
       - Количество нарушений парковки: {count_violations}

    3.  Примечания: 
       - Отсутствуют

    4.  Время обследования: 
       {current_time}

    5.  Инспектор: 
       Данные инспектора"""

    with open('parking_report.txt', 'w') as f:
        f.write(report)
```

### Canny
Canny edge detection is a popular technique used in computer vision for identifying edges in images. It's named after its inventor, John F. Canny, and is widely used due to its accuracy and efficiency. The Canny algorithm involves several steps, including noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding.
Function Description
The Canny function in OpenCV is used to apply the Canny edge detection algorithm to an image. It takes the following parameters:
image: The input image (grayscale) on which edge detection will be performed.
threshold1: The lower threshold for edge detection. Any gradient value below this threshold is considered as a non-edge pixel.
threshold2: The upper threshold for edge detection. Any gradient value above this threshold is considered as an edge pixel.
apertureSize: Optional parameter specifying the aperture size for the Sobel operator (gradient calculation). By default, it's set to 3.
L2gradient: Optional parameter indicating whether to use the 
gradient norm for gradient calculation. If True, it uses the Euclidean distance metric; if False, it uses the 
  norm. By default, it's set to False.
- **Description:** Finds edges in an image using the Canny edge detector.
- **Syntax:** `cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])`
- **Example:** `edges = cv2.Canny(img, 100, 200)`
```python
mark = []
count_violations = 0
count_rights = 0
coordinates_violations = []

print("TakeOff")
set_effect(effect='blink', r=0, g=0, b=255)  # blink with white color
navigate(x=0, y=0, z=1, frame_id='body', auto_arm=True)
rospy.sleep(2)

image_marker = rospy.Subscriber('aruco_detect/markers', MarkerArray, markers_callback)
image_sub = rospy.Subscriber('main_camera/image_raw', Image, image_callback, queue_size=1)
```

## Conclusion
In conclusion, OpenCV provides a comprehensive set of functions for image processing, video processing, and computer vision tasks. From basic operations like reading and displaying images to advanced techniques such as edge detection and feature extraction, OpenCV offers a versatile toolkit for developing robust and efficient computer vision applications.
This guide covers only a subset of the functions available in OpenCV. For a comprehensive list of functions and detailed documentation, refer to the official OpenCV documentation.
```python
main_fly()

create_report()

image_sub.unregister()
image_marker.unregister()
rospy.sleep(1)
print("Land")
land()
print(mark)
```
