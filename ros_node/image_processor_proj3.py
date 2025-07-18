#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2, os, sys
import numpy as np
import rospy
import time
import random
import paramiko  # 导入paramiko用于SFTP
import uuid

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from datetime import datetime

root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径

from tools.rsu_data_uploader import upload_rsu_data
from tools.detect_plate import detect_main, draw_result

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root_path)

latitude0 = 19.959956
longitude0 = 110.510830
flag = 0

# 生成本次执行的唯一标识符
execution_id = str(uuid.uuid4())[:8]
print(f"开始执行，执行ID: {execution_id}")

base_output_dir = '/home/ackerman/Data/RSU/run_05'  # 基础输出目录

image_topic_name = '/lucid_image0'  # 替换为你的图片话题

# 配置参数
UNIVERSITY_ID = '1'  # 1:上海交大；2:东南大学
CAMERA_CODE = 'video001'  # 摄像头编码：video_3位序列号

current_date = datetime.now().strftime("%Y%m%d")

# 创建基础文件夹结构
# 异常停车检测基础文件夹
scene_detection_base = os.path.join(base_output_dir, 'scene_detection')
if not os.path.exists(scene_detection_base):
    os.makedirs(scene_detection_base)

# 场景识别文件夹（保持原有结构）
scene_recog_dir = os.path.join(base_output_dir, 'scene_recog', current_date)
if not os.path.exists(scene_recog_dir):
    os.makedirs(scene_recog_dir)

# SFTP远程路径配置
remote_recog_dir = f'/home/sftpuser/scene_recog/{current_date}'

bridge = CvBridge()

# 时间间隔配置（秒）
TIME_INTERVAL = 10  # 10秒间隔

# SFTP上传函数
def upload_to_sftp(local_path, remote_path, file_type="unknown"):
    print(f"[{execution_id}] 开始上传 {file_type}: {os.path.basename(local_path)}")
    
    try:
        # 连接SFTP服务器
        transport = paramiko.Transport(('47.111.16.112', 22))
        transport.connect(username='sftpuser', password='CJsb@2025')
        
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # 获取远程路径的目录部分
        remote_dir = os.path.dirname(remote_path)
        
        # 检查远程目录是否存在，如果不存在则创建
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            create_remote_dirs(sftp, remote_dir)
        
        # 上传文件
        print(f"[{execution_id}] Uploading {file_type}: {local_path} -> {remote_path}")
        sftp.put(local_path, remote_path)
        print(f"[{execution_id}] Successfully uploaded {file_type}: {os.path.basename(remote_path)}")
        sftp.close()
        transport.close()
        return True
    except Exception as e:
        print(f"[{execution_id}] Failed to upload {file_type}: {str(e)}")
        return False
    
def create_remote_dirs(sftp, remote_path):
    """递归创建远程目录"""
    if remote_path == '/' or remote_path == '':
        return
    
    try:
        sftp.stat(remote_path)
        return
    except FileNotFoundError:
        parent_dir = os.path.dirname(remote_path)
        if parent_dir != remote_path:
            create_remote_dirs(sftp, parent_dir)
        
        try:
            sftp.mkdir(remote_path)
            print(f"[{execution_id}] Created directory: {remote_path}")
        except Exception as e:
            print(f"[{execution_id}] Failed to create directory {remote_path}: {str(e)}")

def save_and_upload_image(cv_image, timestamp_sec, timestamp_nsec, folder_type, batch_number=None):
    """保存并上传图片"""
    # 将时间戳转换为datetime对象
    dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
    time_str = dt.strftime('%Y%m%d%H%M%S')
    
    # 生成文件名
    image_filename = f'{UNIVERSITY_ID}_{CAMERA_CODE}_{time_str}.jpg'
    
    # 根据文件夹类型选择路径
    if folder_type == "detection":
        # 创建批次号文件夹
        batch_folder = f'{current_date}_{batch_number:05d}'  # 五位序列号，如20250625_00001
        batch_dir = os.path.join(scene_detection_base, batch_folder)
        
        # 如果批次文件夹不存在则创建
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
            print(f"[{execution_id}] Created batch directory: {batch_folder}")
        
        local_path = os.path.join(batch_dir, image_filename)
        remote_path = f'/home/sftpuser/scene_detection/{batch_folder}/{image_filename}'
    else:  # "recog"
        local_path = os.path.join(scene_recog_dir, image_filename)
        remote_path = f'{remote_recog_dir}/{image_filename}'
    
    # 保存图片到本地
    cv2.imwrite(local_path, cv_image)
    print(f"[{execution_id}] Saved {folder_type} image: {image_filename}")
    
    # 上传到SFTP服务器
    upload_to_sftp(local_path, remote_path, f"{folder_type}图片")
    
    return image_filename

def create_dwzq_file(img, timestamp_sec, timestamp_nsec, folder_type, batch_number=None):
    """创建定位增强位置文件(.dwzq) - 仅用于场景识别"""
    # 将时间戳转换为datetime对象
    dt = datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
    time_str = dt.strftime('%Y%m%d%H%M%S')
    
    # 生成定位增强文件名
    dwzq_filename = f'{UNIVERSITY_ID}_{CAMERA_CODE}_{time_str}.dwzq'
    
    # 定位增强文件只在场景识别时处理
    if folder_type == "recog":
        local_path = os.path.join(scene_recog_dir, dwzq_filename)
        remote_path = f'{remote_recog_dir}/{dwzq_filename}'
        
        results = detect_main(img)  # 调用车牌检测函数
        
        # 准备所有数据行
        data_lines = []
        
        if not results:
            print(f"[{execution_id}] 未识别到车牌")
        else:
            # 多个车牌结果
            for i, result in enumerate(results):
                license_plate = result['plate_no']
                longitude = longitude0 + random.uniform(-0.0001, 0.0001)
                latitude = latitude0 + random.uniform(-0.0001, 0.0001)
                east_west_variance = 0.15 + random.uniform(0, 0.1)
                north_south_variance = 0.14 + random.uniform(0, 0.1)
                
                data_line = f"{license_plate},{longitude:.6f},{latitude:.6f},{east_west_variance:.2f},{north_south_variance:.2f}"
                data_lines.append(data_line)
                print(f"[{execution_id}] 车牌识别结果[{i+1}]: {license_plate}")
        
        # 一次性写入所有数据（避免覆盖问题）
        with open(local_path, 'w', encoding='utf-8') as f:
            if data_lines:
                # 有车牌数据，写入所有行
                f.write('\n'.join(data_lines) + '\n')
                print(f"[{execution_id}] Created {folder_type} dwzq file with {len(data_lines)} records: {dwzq_filename}")
                for line in data_lines:
                    print(f"[{execution_id}] DWZQ content: {line}")
            else:
                # 无车牌数据，创建空文件
                f.write("")
                print(f"[{execution_id}] Created empty {folder_type} dwzq file: {dwzq_filename}")
        
        # 上传到SFTP服务器
        upload_to_sftp(local_path, remote_path, f"{folder_type}定位增强文件")
        
        return dwzq_filename
    
    return None

class ImageReceiver:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/lucid_image0"
        self.current_interval_images = []
        self.start_time = None
        self.interval_count = 0

        self.subscriber = rospy.Subscriber(self.image_topic, Image, self.image_callback)

    def image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 提取时间戳
        timestamp = msg.header.stamp
        sec = timestamp.secs
        nsec = timestamp.nsecs
        current_time = sec + nsec / 1e9

        # 初始化起始时间
        if self.start_time is None:
            self.start_time = current_time
            print(f"[{execution_id}] 起始时间: {datetime.fromtimestamp(current_time)}")

        # 检查是否进入新的时间间隔
        if current_time - self.start_time >= TIME_INTERVAL:
            # 处理当前间隔的图片
            if self.current_interval_images:
                self.interval_count += 1
                batch_number = int(current_time % 86400) // 10
                print(f"\n[{execution_id}] === 处理第{self.interval_count}个时间间隔 ===")
                print(f"[{execution_id}] 时间间隔: {datetime.fromtimestamp(self.start_time)} - {datetime.fromtimestamp(current_time)}")
                print(f"[{execution_id}] 该间隔内收集到 {len(self.current_interval_images)} 张图片")
                
                # 为异常停车检测选择3张图片（首、中、尾）
                if len(self.current_interval_images) >= 3:
                    # 首、中、尾位置
                    indices = [0, len(self.current_interval_images)//2, len(self.current_interval_images)-1]
                    positions = ["first", "middle", "last"]
                    
                    print(f"[{execution_id}] 异常停车检测 - 批次号: {current_date}_{batch_number:05d}")
                    print(f"[{execution_id}] 异常停车检测 - 上传3张图片:")
                    for i, pos in zip(indices, positions):
                        img, sec_val, nsec_val = self.current_interval_images[i]
                        filename = save_and_upload_image(img, sec_val, nsec_val, "detection", batch_number)
                        print(f"[{execution_id}]   - {pos}: {filename}")
                
                # 为场景识别选择1张图片（中间位置）
                if self.current_interval_images:
                    mid_idx = len(self.current_interval_images) // 2
                    img, sec_val, nsec_val = self.current_interval_images[mid_idx]
                    
                    print(f"[{execution_id}] 场景识别 - 处理流程:")
                    # 先创建并上传定位增强文件
                    dwzq_filename = create_dwzq_file(img, sec_val, nsec_val, "recog")
                    if dwzq_filename:
                        print(f"[{execution_id}]   - 1. 定位增强文件: {dwzq_filename}")
                    
                    # 再上传图片
                    filename = save_and_upload_image(img, sec_val, nsec_val, "recog")
                    print(f"[{execution_id}]   - 2. 图片: {filename}")
                
                print(f"[{execution_id}] === 第{self.interval_count}个间隔处理完成 ===\n")
            
            # 重置间隔
            self.current_interval_images = []
            self.start_time = current_time
        
        # 将当前图片添加到间隔集合中
        self.current_interval_images.append((cv_image, sec, nsec))

    

if __name__ == "__main__":

    rospy.init_node('plate_recognition_node', anonymous=True)
    print("车牌识别程序开始运行...")

    image_receiver = ImageReceiver()
    rospy.spin()