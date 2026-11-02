cmake_minimum_required(VERSION 3.16)
project(dwarl_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

include(cmake/onnxruntime.cmake)

# ROS-free core: deploy.yaml contract, observation building, ONNX inference
add_library(dwarl_policy_core SHARED src/policy_core.cpp)
target_include_directories(dwarl_policy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(dwarl_policy_core PUBLIC onnxruntime::onnxruntime PRIVATE yaml-cpp::yaml-cpp)
set_target_properties(dwarl_policy_core PROPERTIES INSTALL_RPATH "$ORIGIN;${ONNXRUNTIME_RPATH}")

# Nav2 controller plugin
add_library(dwarl_controller SHARED src/controller.cpp src/profiled_controller.cpp)
target_link_libraries(dwarl_controller PUBLIC dwarl_policy_core
  ${geometry_msgs_TARGETS} ${diagnostic_msgs_TARGETS} ${nav2_msgs_TARGETS} ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS} ${visualization_msgs_TARGETS}
  nav2_costmap_2d::nav2_costmap_2d_core nav2_util::nav2_util_core pluginlib::pluginlib
  rclcpp::rclcpp rclcpp_lifecycle::rclcpp_lifecycle tf2::tf2 tf2_geometry_msgs::tf2_geometry_msgs tf2_ros::tf2_ros)
set_target_properties(dwarl_controller PROPERTIES INSTALL_RPATH "$ORIGIN;${ONNXRUNTIME_RPATH}")
target_include_directories(dwarl_controller PUBLIC ${nav2_core_INCLUDE_DIRS})
pluginlib_export_plugin_description_file(nav2_core dwarl_controller_plugin.xml)

install(TARGETS dwarl_policy_core dwarl_controller
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
if(ONNXRUNTIME_BUNDLED)
  # ship the downloaded runtime next to the plugin (found through $ORIGIN rpath)
  file(GLOB _ort_libs "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
  install(FILES ${_ort_libs} DESTINATION lib)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(nlohmann_json REQUIRED)
  # DWARL repo root (ros/src/pkgs/<pkg> -> ../../../..): the golden test uses the latest exported run
  get_filename_component(DWARL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." REALPATH)
  ament_add_gtest(test_policy_core test/test_policy_core.cpp)
  target_link_libraries(test_policy_core dwarl_policy_core nlohmann_json::nlohmann_json)
  target_compile_definitions(test_policy_core PRIVATE DWARL_REPO_ROOT="${DWARL_REPO_ROOT}")
endif()

ament_export_include_directories(include/${PROJECT_NAME})
ament_export_dependencies(nav2_core nav2_costmap_2d nav2_util pluginlib rclcpp rclcpp_lifecycle
  sensor_msgs diagnostic_msgs tf2_ros visualization_msgs)
ament_package()
