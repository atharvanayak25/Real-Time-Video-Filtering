# ATHARVA NAYAK #
# NU ID: 002322653 #

cmake_minimum_required(VERSION 3.10)
project(vidDisplay VERSION 0.1.0 LANGUAGES CXX)



# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Add the executable and link the libraries
add_executable(vidDisplay vidDisplay.cpp filter.cpp faceDetect.cpp ExtensionaddCaption.cpp ExtensionaddFrame.cpp)
add_executable(imgDisplay imgDisplay.cpp )

target_link_libraries(vidDisplay PRIVATE ${OpenCV_LIBS})
target_link_libraries(imgDisplay PRIVATE ${OpenCV_LIBS})

# Enable testing (optional)
include(CTest)
enable_testing()

# Set project name and version for packaging
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
