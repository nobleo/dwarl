cmake_minimum_required(VERSION 3.16)
project(dwarl_bringup NONE)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# DWARL_ROOT (repo containing runs/ and data/) defaults to this source tree; override by exporting it.
get_filename_component(DWARL_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." REALPATH)
ament_environment_hooks(hooks/dwarl_root.dsv.in)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS scripts/policy_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY behavior_trees config launch DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_nav_params test/test_nav_params.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})
  ament_add_pytest_test(test_behavior_tree test/test_behavior_tree.py)
endif()

ament_package()
