cmake_minimum_required(VERSION 3.0.2)
project(efficientdet_deploy)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
  cv_bridge
  efficientdet_ros_msgs
)



#find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    std_msgs
    efficientdet_ros_msgs
    image_transport
  DEPENDS
    Boost
)

set(TENSORFLOW_LIB_DIR "/opt/tensorflow/lib") 
###########
## Build ##
###########

include_directories(
                    include
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    )


add_executable(efficientdet_node
  
  src/efficientdet_handler.cpp
  src/efficientdet_node.cpp
  include/efficientdet_deploy/NetTf.cpp
  include/efficientdet_deploy/utils.cpp
)

#target_include_directories(efficientdet_node PUBLIC include)
target_include_directories(efficientdet_node PRIVATE ${TENSORFLOW_LIB_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(efficientdet_node ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} )

# add_dependencies(
#   efficientdet_ros_msgs_generate_messages_cpp
# )
