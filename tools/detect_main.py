# -*- coding: UTF-8 -*-
import os, sys
import cv2

from detect_plate import detect_main


if __name__ == "__main__":
    
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # tools/..
    os.chdir(root_path)

    img_path = "/home/ackerman/Data/RSU/run_04/scene_detection/20250625_00001/1_video001_20250625175312.jpg"
    img = cv2.imread(img_path)  # 读取图像
    result_dicts = detect_main(img, save_path="./result/", img_path=img_path)  

    for result_dict in result_dicts:
        print("车牌号:", result_dict['plate_no'])
        print("车牌角点坐标:", result_dict['landmarks'])
        print("车牌检测得分:", result_dict['detect_conf'])
        print("每个字符的概率:", result_dict['rec_conf'])
        print("车牌颜色:", result_dict.get('plate_color', '未知'))