#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2, os
import numpy as np

import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import time

from v2x_msgs.msg import RoadCarMsg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion

from tools.rsu_data_uploader import upload_rsu_data
from tools.detect_plate import detect_main, draw_result

os.chdir('/home/ackerman/Workspace/plate_recognition_ws/src/plate_recognition')

latitude = 19.959956
longitude = 110.510830
height_global = 80.0  # 假设高度为100米
yaw = 0.0  # 假设初始方向角为0度

# 筛选出点云里落在车牌内的点
def is_point_inside_plate(point, corner_points):
    # 使用向量叉积判断点是否在多边形内
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    p = np.array(point)
    c1 = np.array(corner_points[0])
    c2 = np.array(corner_points[1])
    c3 = np.array(corner_points[2])
    c4 = np.array(corner_points[3])

    # 计算向量
    v1 = c2 - c1
    v2 = p - c1
    v3 = c3 - c2
    v4 = p - c2
    v5 = c4 - c3
    v6 = p - c3
    v7 = c1 - c4
    v8 = p - c4

    # 计算叉积
    cp1 = cross_product(v1, v2)
    cp2 = cross_product(v3, v4)
    cp3 = cross_product(v5, v6)
    cp4 = cross_product(v7, v8)

    return (cp1 >= 0 and cp2 >= 0 and cp3 >= 0 and cp4 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0 and cp4 <= 0)

def project_points_to_image(points, K):
    """
    Projects 3D points to 2D image coordinates using the camera intrinsic matrix K.
    """
    # points: (N, 3), K: (3, 3)
    projected_points = K @ points.T  # (3, N)
    projected_points /= projected_points[2, :]  # Normalize by z coordinate
    # nan values can occur if z is zero, we can filter them out
    valid_mask = ~np.isnan(projected_points[0, :]) & ~np.isnan(projected_points[1, :]) & (projected_points[2, :] > 0)
    projected_points = projected_points[:, valid_mask]  # Filter out invalid points
    return projected_points.T  # (N, 3)

# 把plate_points_3d拟合为平面
def fit_plane(points):
    """
    使用最小二乘法拟合平面 ax + by + cz + d = 0
    返回平面参数 (a, b, c, d)
    """
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    a, b, d = C
    c = -1
    return a, b, c, d

# 然后把plate_points_2d的中心，也就是像素坐标投影到3d平面上，计算3d坐标
def project_to_plane(points_2d, plane_params, K):
    """
    将2D点投影到3D平面上
    points_2d: (N, 2) 像素坐标
    plane_params: (a, b, c, d) 平面方程参数 ax + by + cz + d = 0
    K: 相机内参矩阵
    返回投影后的3D点 (N, 3)
    """
    a, b, c, d = plane_params
    K_inv = np.linalg.inv(K)
    points_3d = []
    
    for point in points_2d:
        u, v = point
        # 将像素坐标转换为齐次坐标
        pixel_ho_mo = np.array([u, v, 1])
        # 使用内参矩阵的逆将像素坐标转换为归一化坐标
        normalized = K_inv @ pixel_ho_mo
        x_norm, y_norm = normalized[0], normalized[1]
        
        # 根据平面方程求解深度z: ax + by + cz + d = 0
        # 其中 x = z * x_norm, y = z * y_norm
        # 所以 a(z*x_norm) + b(z*y_norm) + cz + d = 0
        # z(a*x_norm + b*y_norm + c) = -d
        z = -d / (a * x_norm + b * y_norm + c)
        
        # 计算3D坐标
        x = z * x_norm
        y = z * y_norm
        points_3d.append([x, y, z])
    
    return np.array(points_3d)

def lan_lon_add_x_y(lat, lon, x, y, yaw):
    """
    根据经纬度和偏移量计算新的经纬度
    lat: 纬度
    lon: 经度
    x: 雷达系x
    y: 雷达系y
    yaw: 雷达系方向角, x轴 北偏东
    返回新的经纬度 (latitude, longitude)
    """
    R = 6378137.0  # 地球半径，单位为米
    dE = x * np.cos(np.radians(yaw)) - y * np.sin(np.radians(yaw))  # 东向偏移
    dN = x * np.sin(np.radians(yaw)) + y * np.cos(np.radians(yaw))  # 北向偏移 

    d_lat = dN / R  # 纬度变化量
    d_lon = dE / (R * np.cos(np.pi * lat / 180))  # 经度变化量

    new_lat = lat + d_lat * (180 / np.pi)
    new_lon = lon + d_lon * (180 / np.pi)

    return new_lat, new_lon

class ImageLidarAligner:
    def __init__(self):
        self.bridge = CvBridge()
        self.recent_plates = []  # 保存最近两个图像的车牌信息 [(timestamp, corner_points, plate_no, img_draw), ...]
        self.max_plate_history = 2  # 最多保存2个图像的信息

        rospy.Subscriber("/lucid_image1", Image, self.image_callback)
        rospy.Subscriber("/livox/lidar", PointCloud2, self.lidar_callback)
        self.v2x_pub = rospy.Publisher('/road_car_topic', RoadCarMsg, queue_size=10)
        self.plate_recognition_publisher = rospy.Publisher('/plate_recognition_result', Image, queue_size=10)
        self.plate_marker_publisher = rospy.Publisher('/plate_marker', Marker, queue_size=10)

    def image_callback(self, msg):
        """处理图像消息，检测车牌并保存最近两个检测结果"""
        timestamp_image = msg.header.stamp.to_sec()
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 进行车牌检测
        result_dicts = detect_main(img)
        
        if result_dicts is not None and len(result_dicts) > 0:
            result_dict = result_dicts[0]
            corner_points = result_dict['landmarks']
            plate_no = result_dict['plate_no']
            img_draw = draw_result(img, [result_dict])
            
            print(f"检测到车牌号: {plate_no}, 时间戳: {timestamp_image:.3f}")
            print(f"车牌角点坐标: {corner_points}")
            
            # 保存检测结果
            plate_info = {
                'timestamp': timestamp_image,
                'corner_points': corner_points,
                'plate_no': plate_no,
                'img_draw': img_draw
            }
            
            self.recent_plates.append(plate_info)
            
            # 只保留最近两个检测结果
            if len(self.recent_plates) > self.max_plate_history:
                self.recent_plates.pop(0)
            
            # 发布检测结果图像
            img_msg_out = self.bridge.cv2_to_imgmsg(img_draw, encoding='bgr8')
            self.plate_recognition_publisher.publish(img_msg_out)
        else:
            print("未检测到车牌")
            # 发布原始图像
            self.plate_recognition_publisher.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

    def lidar_callback(self, msg):
        """处理点云消息，通过插值估计对应时刻的车牌corner点"""
        lidar_timestamp = msg.header.stamp.to_sec()
        
        # 需要至少有一个车牌检测结果才能进行处理
        if len(self.recent_plates) == 0:
            return
        
        # 获取插值后的车牌corner点
        interpolated_corner = self.interpolate_plate_corner(lidar_timestamp)
        
        if interpolated_corner is not None:
            print(f"点云时间戳: {lidar_timestamp:.3f}")
            print(f"插值后的车牌角点: {interpolated_corner}")
            
            # 使用插值后的corner点处理点云数据
            result_dict = {'landmarks': interpolated_corner}
            center_3d = self.process_lidar_data(msg, result_dict)
            
            # 如果成功计算出3D位置，发布坐标轴标记
            if center_3d is not None:
                center_3d_for_marker = np.array([center_3d[2], -center_3d[0], -center_3d[1]])
                print(f"车牌中心3D坐标用于标记: {center_3d_for_marker}")
                marker = self.create_coordinate_marker(center_3d_for_marker, "plate")
                self.plate_marker_publisher.publish(marker)

            if center_3d is not None:
                # publish RSU数据
                road_car_msg = RoadCarMsg()
                # ID字段需要是8个uint8的数组，将字符串转换为字节数组
                plate_no = self.recent_plates[-1]['plate_no']
                plate_no[0] = '-'
                print(f"KKKKKKKKKKKKKKKKK发布车牌号: {plate_no}")
                # 将字符串转换为字节，然后填充或截断到8字节
                id_bytes = plate_no.encode('utf-8')[:8]  # 取前8个字节
                id_array = list(id_bytes) + [0] * (8 - len(id_bytes))  # 如果不足8字节则用0填充
                road_car_msg.ID = id_array

                center_3d_in_lidar = np.array([center_3d[2], -center_3d[0], -center_3d[1]])
                # 将3D坐标转换为经纬度
                latitude, longitude = lan_lon_add_x_y(latitude, longitude, center_3d_in_lidar[0], center_3d_in_lidar[1], yaw)
                print(f"转换后的经纬度: ({latitude}, {longitude})")
                # 填充RSU数据
                road_car_msg.header.stamp = lidar_timestamp

                road_car_msg.latitude = int(latitude * 1e9)  # 转换为整数
                road_car_msg.longitude = int(longitude * 1e9)  # 转换为整数
                road_car_msg.height = int((height_global + center_3d_in_lidar[2]) * 1e5)  # 假设高度为100米
                print(f"发布车牌号: {plate_no}, 经纬度: ({road_car_msg.latitude}, {road_car_msg.longitude}), 高度: {road_car_msg.height}")

                road_car_msg.secs = int(lidar_timestamp)
                road_car_msg.nsecs = int((lidar_timestamp - road_car_msg.secs) * 1e9)

                print(f"发布时间戳: ：：：：：：：：{road_car_msg.secs}.{road_car_msg.nsecs}")
                      
                road_car_msg.accident_tp = 10
                # 发布消息
                self.v2x_pub.publish(road_car_msg)
            
        else:
            print(f"无法为点云时间戳 {lidar_timestamp:.3f} 进行插值")

    def interpolate_plate_corner(self, target_timestamp):
        """
        根据目标时间戳插值计算车牌corner点
        如果只有一个检测结果，直接返回该结果
        如果有两个检测结果，进行线性插值
        """
        if len(self.recent_plates) == 1:
            # 只有一个检测结果，检查时间差是否在合理范围内
            plate_info = self.recent_plates[0]
            time_diff = abs(target_timestamp - plate_info['timestamp'])
            if time_diff <= 0.5:  # 最大允许0.5秒的时间差
                return plate_info['corner_points']
            else:
                return None
        
        elif len(self.recent_plates) == 2:
            # 有两个检测结果，进行线性插值
            plate1, plate2 = self.recent_plates[0], self.recent_plates[1]
            t1, t2 = plate1['timestamp'], plate2['timestamp']
            
            # 确保时间顺序
            if t1 > t2:
                plate1, plate2 = plate2, plate1
                t1, t2 = t2, t1
            
            # 检查目标时间是否在两个检测时间之间或附近
            if target_timestamp < t1 - 0.2 or target_timestamp > t2 + 0.2:
                # 时间差太大，使用最近的一个
                if abs(target_timestamp - t1) < abs(target_timestamp - t2):
                    return plate1['corner_points']
                else:
                    return plate2['corner_points']
            
            # 进行线性插值
            if t2 - t1 < 1e-6:  # 避免除零
                return plate2['corner_points']
            
            # 插值系数
            alpha = (target_timestamp - t1) / (t2 - t1)
            alpha = max(0, min(1, alpha))  # 限制在[0,1]范围内
            
            # 对每个角点进行插值
            corner1 = np.array(plate1['corner_points'])
            corner2 = np.array(plate2['corner_points'])
            interpolated_corner = corner1 * (1 - alpha) + corner2 * alpha
            
            return interpolated_corner.tolist()
        
        return None

    def process_lidar_data(self, lidar_msg, result_dict):

        # lidar msg to points
        pcd = []
        for p in pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True):
            pcd.append(p)
        pcd = np.array(pcd)
        if pcd.size == 0:
            print("未获取到有效的点云数据")
            return None, None
        
        # 处理点云数据
        f = 25 * 1e-3  # focal length in meters
        sx = 3.45 * 1e-6 
        sy = 3.45 * 1e-6  # focal length in meters
        cx = 1224
        cy = 1024
        K = np.array([[f/sx, 0, cx],
                    [0, f/sy, cy],
                    [0, 0, 1]])

        # Define the rotation matrix to convert from LiDAR frame to camera frame
        R_CL = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]])
        t_CL = np.array([-0.05, -0.1, -0.05])  # No translation in this example

        # Transform the point cloud from LiDAR frame to camera frame
        pcd_points = pcd @ R_CL.T  # Apply rotation to point cloud
        pcd_points += t_CL  # Apply translation to point cloud

        projected_points = project_points_to_image(pcd_points, K)
        # Compute valid_mask for projected points within image bounds
        valid_mask = (
            (projected_points[:, 0] >= 0) & (projected_points[:, 0] < 1224) &
            (projected_points[:, 1] >= 0) & (projected_points[:, 1] < 1024)
        )
        valid_points_2d = projected_points[valid_mask, :2]  # 只保留2D坐标
        # 记录2D点对应的3D点坐标
        valid_points_3d = pcd_points[valid_mask]

        corner_points = result_dict['landmarks']
        print("车牌角点坐标:", corner_points)
        # 左上，右上， 右下， 左下

        # 筛选出车牌区域内的点云
        plate_points_3d = []
        plate_points_2d = []
        for i, point_2d in enumerate(valid_points_2d):
            if is_point_inside_plate(point_2d[:2], corner_points):
                plate_points_3d.append(valid_points_3d[i])
                plate_points_2d.append(point_2d)
        
        if len(plate_points_3d) > 0:
            plate_points_3d = np.array(plate_points_3d)
            plane_params = fit_plane(plate_points_3d)
            print(f"车牌平面参数: {plane_params}")

            # 计算车牌中心的2D位置
            corner_points = np.array(corner_points)
            center_2d = np.mean(corner_points, axis=0)
            # 将2D点投影到3D平面上
            center_3d = project_to_plane([center_2d], plane_params, K)
            center_3d = center_3d[0]  # 取第一个
            print(f"****车牌中心3D坐标: {center_3d}")
            
            return center_3d
        else:
            print("车牌区域内没有找到点云数据")
            return None

    def create_coordinate_marker(self, center_3d, frame_id="plate"):
        """
        创建表示车牌位置的坐标轴箭头标记
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "plate_coordinates"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 设置箭头位置 (起点)
        start_point = Point()
        start_point.x = center_3d[0]
        start_point.y = center_3d[1] 
        start_point.z = center_3d[2]
        
        # 设置箭头终点 (指向Z轴正方向，表示车牌法向量)
        end_point = Point()
        end_point.x = center_3d[0]
        end_point.y = center_3d[1]
        end_point.z = center_3d[2] + 0.5  # 箭头长度50cm
        
        marker.points = [start_point, end_point]
        
        # 设置箭头尺寸
        marker.scale.x = 0.5  # 箭头轴的直径
        marker.scale.y = 1   # 箭头头部的直径
        marker.scale.z = 1   # 箭头头部的长度
        
        # 设置颜色 (红色)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rospy.Duration(1.0)  # 标记持续1秒
        
        return marker


if __name__ == "__main__":

    rospy.init_node('plate_recognition_node', anonymous=True)
    print("车牌识别程序开始运行...")

    aligner = ImageLidarAligner()
    rospy.spin()

    # for _ in range(100):
    #     # 创建一个road2carmsg
    #     road_car_msg = RoadCarMsg()
        
    #     # ID字段需要是8个uint8的数组，将字符串转换为字节数组
    #     plate_no = "-ABCDF2"
    #     # 将字符串转换为字节，然后填充或截断到8字节
    #     id_bytes = plate_no.encode('utf-8')[:8]  # 取前8个字节
    #     id_array = list(id_bytes) + [0] * (8 - len(id_bytes))  # 如果不足8字节则用0填充
    #     road_car_msg.ID = id_array
        
    #     road_car_msg.latitude = int(latitude * 1e9)  # 转换为整数
    #     road_car_msg.longitude = int(longitude * 1e9)  # 转换为整数
    #     road_car_msg.secs = int(time.time())
    #     road_car_msg.nsecs = int((time.time() - road_car_msg.secs) * 1e9)
    #     road_car_msg.height = 100
    #     road_car_msg.accident_tp = 0

    #     # 发布消息
    #     publisher = rospy.Publisher('/road_car_topic', RoadCarMsg, queue_size=10)
    #     publisher.publish(road_car_msg)

    #     rospy.sleep(1)  # 等待1秒钟

