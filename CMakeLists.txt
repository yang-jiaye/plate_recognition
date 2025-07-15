cmake_minimum_required(VERSION 3.0.2)
project(plate_recognition)

find_package(catkin REQUIRED COMPONENTS
  rospy
  v2x_msgs
  std_msgs
  cv_bridge 
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# catkin_install_python(PROGRAMS
#   src/rsu_data_uploader/rsu_data_uploader.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )