cmake_minimum_required(VERSION 3.12)
project(vision_cpp)

# 1. Standard C++ for robotics (C++17 is ideal for GTSAM and modern CV)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Find Python and its development headers
# This ensures we build for the Python you are currently using
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 3. Find pybind11
# If you installed it via 'pip install pybind11', this line finds it
find_package(pybind11 CONFIG REQUIRED)

# 4. Find OpenCV
find_package(OpenCV REQUIRED)

# 5. Create the module
# This combines your .cpp into a Python-loadable .so file
pybind11_add_module(vision_cpp vision_module.cpp)

# 6. Link OpenCV
target_link_libraries(vision_cpp PRIVATE ${OpenCV_LIBS})

# 7. Move the output to your Python folder automatically
set_target_properties(vision_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../core"
    PREFIX "" # Removes 'lib' from 'libvision_cpp.so'
)

message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Using Python: ${Python_EXECUTABLE}")