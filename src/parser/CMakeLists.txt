project("sql_parser")

# execute_process(COMMAND ./generate_parser.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE
        parser_files
        CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

list(FILTER parser_files EXCLUDE REGEX .*main.cpp)

add_library(sql_parser
        STATIC
        ${parser_files}
)

# Link against infinity_core to access its modules
target_link_libraries(sql_parser PRIVATE infinity_core)

target_include_directories(sql_parser PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(sql_parser PUBLIC "${CMAKE_SOURCE_DIR}/third_party/spdlog/include")
target_include_directories(sql_parser PUBLIC "${CMAKE_SOURCE_DIR}/third_party/nlohmann")
target_include_directories(sql_parser PUBLIC "${CMAKE_SOURCE_DIR}/third_party/simdjson")
target_include_directories(sql_parser PUBLIC "${CMAKE_SOURCE_DIR}/third_party/parallel-hashmap")
target_include_directories(sql_parser PUBLIC "${CMAKE_SOURCE_DIR}/third_party/arrow/src")
