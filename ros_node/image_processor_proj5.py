#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2, os, sys
import numpy as np
import rospy
import time
import random

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径

from tools.rsu_data_uploader import upload_rsu_data
from tools.detect_plate import detect_main, draw_result

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_path)

latitude = 19.959956
longitude = 110.510830
flag = 1

class ImageReceiver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/lucid_image0"
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.publisher = rospy.Publisher('/plate_recognition_result', Image, queue_size=10)

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        result_dicts = detect_main(img)

        if result_dicts is None:
            print("未检测到车牌，请检查图像质量或车牌位置。")
            self.publisher.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        else:
            for result_dict in result_dicts:
                print("车牌号:", result_dict['plate_no'])
                print("车牌角点坐标:", result_dict['landmarks'])
                print("车牌检测得分:", result_dict['detect_conf'])
                print("每个字符的概率:", result_dict['rec_conf'])
                print("车牌颜色:", result_dict.get('plate_color', '未知'))

                upload_rsu_data(latitude + 1e-5 * random.random(), longitude + 1e-5 * random.random(), result_dict['plate_no'], 1)

            img_draw = draw_result(img, result_dicts)
            img_msg = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            self.publisher.publish(img_msg)

if __name__ == "__main__":

    rospy.init_node('plate_recognition_node', anonymous=True)
    print("车牌识别程序开始运行...")

    image_receiver = ImageReceiver()
    rospy.spin()