cmake_minimum_required(VERSION 3.0.2)
project(ovir_3d)

find_package(catkin REQUIRED)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES yolact_ros
  # CATKIN_DEPENDS
  #  DEPENDS system_lib
)

install(
  PROGRAMS scripts/save_camera_datasets.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
