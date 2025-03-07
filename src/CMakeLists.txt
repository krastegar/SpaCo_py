cmake_minimum_required(VERSION 3.14)
project(imhoff)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
add_subdirectory(pybind11 REQUIRED)

# Add the target for the Pybind11 module
pybind11_add_module(imhoff_prob imhoff_prob.cpp)

# Include directories for pybind11
target_include_directories(imhoff_prob PRIVATE ${pybind11_INCLUDE_DIRS})

# Link libraries (if any)
# target_link_libraries(imhoff PRIVATE <your-libraries>)