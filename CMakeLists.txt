cmake_minimum_required(VERSION 2.8.3)
project(yolo_triangulation)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  sensor_msgs
  darknet_ros_msgs
  nav_msgs
  cmake_modules
  message_generation
  mg_msgs
  std_srvs
)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)

catkin_package(
  DEPENDS Eigen
  CATKIN_DEPENDS darknet_ros_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)


add_executable(yolo_triangulation src/triangulation_node.cpp
                                  src/msg_conversions.cpp
                                  src/visualization_functions.cpp)
# add_dependencies(yolo_triangulation yolo_triangulation_generate_messages_cpp)
add_dependencies(yolo_triangulation ${catkin_EXPORTED_TARGETS})
add_dependencies(yolo_triangulation darknet_ros_msgs_gencpp)
target_link_libraries(yolo_triangulation ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})