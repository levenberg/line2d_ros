cmake_minimum_required(VERSION 2.8.3)
project(ulsd)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  lines2d_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)