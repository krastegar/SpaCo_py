cmake_minimum_required(VERSION 3.14)
project(imhoff)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
add_subdirectory(pybind11)

# Find Boost
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.83.0 REQUIRED COMPONENTS system) # system if we don't know exactly where the config file is

# Add the target for the Pybind11 module
pybind11_add_module(imhoff imhoff.cpp)

# Include directories for pybind11 and Boost
target_include_directories(imhoff PRIVATE ${pybind11_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Link libraries (if any)
# target_link_libraries(imhoff PRIVATE ${Boost_LIBRARIES} /home/krastegar0/SpaCo_py/src/quadpack/libquadpack.a)