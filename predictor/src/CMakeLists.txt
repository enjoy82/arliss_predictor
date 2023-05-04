file(GLOB_RECURSE SRCS
    *.cpp)
add_library(predictor STATIC ${SRCS})
target_include_directories(predictor PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(predictor PUBLIC ${PROJECT_SOURCE_DIR}/..)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

set(ov_link_libraries openvino::runtime)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(predictor ${OpenCV_LIBS} ${ov_link_libraries})