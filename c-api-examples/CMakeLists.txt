include_directories(${CMAKE_SOURCE_DIR})

add_executable(decode-file-c-api decode-file-c-api.c)
target_link_libraries(decode-file-c-api sherpa-ncnn-c-api)

add_executable(vad-c-api vad-c-api.c)
target_link_libraries(vad-c-api sherpa-ncnn-c-api)
