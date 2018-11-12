cmake_minimum_required(VERSION 2.8.3)
project(tl_detector)

EXECUTE_PROCESS(
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} ${PROTO_FLAGS} --python_out=${CMAKE_CURRENT_SOURCE_DIR} string_int_label_map.proto
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

message(STATUS "Generated new ${CMAKE_CURRENT_SOURCE_DIR}/string_int_label_map_p2.py")
