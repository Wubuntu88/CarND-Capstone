import cv2
import numpy as np
import time
import rospy
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    # Hue scale: https://colorlitelens.com/color-games.htmll
    def __init__(self):
        RED_MIN_AS_HUE = 0.0 / 360
        RED_MAX_AS_HUE = 20.0 / 360
        
        YELLOW_MIN_AS_HUE = 40.0 / 360
        YELLOW_MAX_AS_HUE = 75.0 / 360
        
        GREEN_MIN_AS_HUE = 75.0 / 360
        GREEN_MAX_AS_HUE = 165.0 / 360
        
        self.RED_MIN_THRESHOLD = np.array([RED_MIN_AS_HUE, 100, 100], np.uint8)
        self.RED_MAX_THRESHOLD = np.array([RED_MAX_AS_HUE, 255, 255], np.uint8)
        
        self.YELLOW_MIN_THRESHOLD = np.array([YELLOW_MIN_AS_HUE, 100, 100], np.uint8)
        self.YELLOW_MAX_THRESHOLD = np.array([YELLOW_MAX_AS_HUE, 255, 255], np.uint8)
        
        self.GREEN_MIN_THRESHOLD = np.array([GREEN_MIN_AS_HUE, 100, 100], np.uint8)
        self.GREEN_MAX_THRESHOLD = np.array([GREEN_MAX_AS_HUE, 255, 255], np.uint8)
        
        self.PIXEL_COUNT_THRESHOLD = 80

    def get_classification(self, rgb_image):
        """ Classifies the image as containing a red, yellow, or green light
        @param rgb_image (cv::Mat): the image that may have a light in it
        @return (int): Traffic light color from styx_msgs/TrafficLight
        """
        start_time_seconds = time.time()
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        
        red_thresholded_image = cv2.inRange(hsv_image, self.RED_MIN_THRESHOLD, self.RED_MAX_THRESHOLD)
        red_pixel_count = cv2.countNonZero(red_thresholded_image)
        if red_pixel_count > self.PIXEL_COUNT_THRESHOLD:
            rospy.loginfo("Detected Red in " + str(time.time() - start_time_seconds) + " seconds")
            return TrafficLight.RED
        
        yellow_thresholded_image = cv2.inRange(hsv_image, self.YELLOW_MIN_THRESHOLD, self.YELLOW_MAX_THRESHOLD)
        yellow_pixel_count = cv2.countNonZero(yellow_thresholded_image)
        if yellow_pixel_count > self.PIXEL_COUNT_THRESHOLD:
            rospy.loginfo("Detected Yellow in " + str(time.time() - start_time_seconds) + " seconds")
            return TrafficLight.YELLOW
        
        green_thresholded_image = cv2.inRange(hsv_image, self.GREEN_MIN_THRESHOLD, self.GREEN_MAX_THRESHOLD)
        green_pixel_count = cv2.countNonZero(green_thresholded_image)
        if green_pixel_count > self.PIXEL_COUNT_THRESHOLD:
            rospy.loginfo("Detected Green in " + str(time.time() - start_time_seconds) + " seconds")
            return TrafficLight.GREEN
        
        return TrafficLight.UNKNOWN









