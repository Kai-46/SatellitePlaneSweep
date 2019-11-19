link_libraries(PSL ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(satellitePinholePlaneSweep satellitePlaneSweep.cpp)
