#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

root_path=os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # 项目根路径：获取当前路径，再上级路径
sys.path.append(root_path)  # 将项目根路径写入系统路径
from tools.rsu_data_uploader import upload_rsu_data

latitude = 19.959956
longitude = 110.510830
flag = 0

if __name__ == '__main__':
    
    # Initialize the uploader
    uploader = upload_rsu_data(latitude, longitude, "苏A5V12M", flag)

