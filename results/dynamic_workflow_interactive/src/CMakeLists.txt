# Set the Target Name and Add Executable
set(TARGET_NAME task1)

# Determine source files based on data type and version
if(TEST_DATA_TYPE STREQUAL "float32")
    set(DATA_TYPE_DIR "matmul_f32")
elseif(TEST_DATA_TYPE STREQUAL "float16")
    set(DATA_TYPE_DIR "matmul_f16")
else()
    message(FATAL_ERROR "Unsupported TEST_DATA_TYPE: ${TEST_DATA_TYPE}")
endif()

# Always include main.cpp
set(CXX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set(CUDA_SOURCES "")

# Compile ONLY the files this binary actually links:
#   main.cpp (above) + the cBLAS reference v0 (TestData uses it as ground truth)
#   + the single selected version (MATMUL_VERSION, the kernel under test).
# main.cpp only ever references matmul<DataType, MATMUL_VERSION> and the v0
# reference, so compiling every sibling version (the old `file(GLOB *)`) was pure
# overhead AND made one broken file block all versions. Selective compile makes
# version builds / small edits fast and isolates broken siblings.
# NOTE: CMake re-globs only on (re)configure; build-task1.sh reconfigures when
#   --ver changes. If you add a 2nd file to the *same* version, reconfigure once
#   (touch this CMakeLists or `./task1.sh clean`).
set(_VDIR "${CMAKE_CURRENT_SOURCE_DIR}/${DATA_TYPE_DIR}")
# v0 = cBLAS ground-truth reference (always linked, used by TestData).
file(GLOB V0_FILES  "${_VDIR}/*_v0_*.cpp" "${_VDIR}/*_v0.cpp"
                    "${_VDIR}/*_v0_*.cu"  "${_VDIR}/*_v0.cu")
# Selected version. Match `_vN_` and `_vN.` so e.g. v2 never matches v20/v22.
file(GLOB SEL_FILES "${_VDIR}/*_v${MATMUL_VERSION}_*.cpp" "${_VDIR}/*_v${MATMUL_VERSION}.cpp"
                    "${_VDIR}/*_v${MATMUL_VERSION}_*.cu"  "${_VDIR}/*_v${MATMUL_VERSION}.cu")

if(NOT SEL_FILES AND NOT MATMUL_VERSION EQUAL 0)
    message(FATAL_ERROR
        "No source for ${DATA_TYPE_DIR} version ${MATMUL_VERSION} "
        "(expected e.g. matmul_..._v${MATMUL_VERSION}_*.cu). "
        "Create the file and name it with _v${MATMUL_VERSION}_.")
endif()

set(SELECTED_FILES ${V0_FILES} ${SEL_FILES})
list(REMOVE_DUPLICATES SELECTED_FILES)   # MATMUL_VERSION==0 -> v0 only, once
foreach(_f ${SELECTED_FILES})
    if(_f MATCHES "\\.cu$")
        list(APPEND CUDA_SOURCES ${_f})
    else()
        list(APPEND CXX_SOURCES ${_f})
    endif()
endforeach()

# Log included files for debugging
log_info("Including source files for ${TEST_DATA_TYPE} version ${MATMUL_VERSION}:")
foreach(source ${CXX_SOURCES})
    log_info("  CXX: ${source}")
endforeach()
foreach(source ${CUDA_SOURCES})
    log_info("  CUDA: ${source}")
endforeach()

add_executable(${TARGET_NAME} ${CXX_SOURCES} ${CUDA_SOURCES})

# [NOTE] Uncomment this to generate preprocessed file and assembly file
# target_compile_options(${TARGET_NAME} PRIVATE -save-temps)

log_info("Matmul Version: ${MATMUL_VERSION}")
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    MATMUL_VERSION=${MATMUL_VERSION}
)

log_info("Test Data Type: ${TEST_DATA_TYPE}")
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    $<IF:$<STREQUAL:${TEST_DATA_TYPE},float32>,TEST_FLOAT32,
    $<IF:$<STREQUAL:${TEST_DATA_TYPE},float16>,TEST_FLOAT16,
    $<ERROR:Unsupported data type: ${TEST_DATA_TYPE}>>>
)

target_include_directories(
    ${TARGET_NAME} PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(
    ${TARGET_NAME} PRIVATE
    CUDA::cudart
    OpenBLAS::OpenBLAS
    cxxopts::cxxopts
)

set(TARGET_BIN_OUTPUT_NAME "${TARGET_NAME}_${TEST_DATA_TYPE}_v${MATMUL_VERSION}")
set_target_properties(
    ${TARGET_NAME} PROPERTIES 
    OUTPUT_NAME 
        ${TARGET_BIN_OUTPUT_NAME}
)

# Add TARGET_BIN_OUTPUT_NAME to compile definitions
target_compile_definitions(
    ${TARGET_NAME} PRIVATE 
    TARGET_BIN_OUTPUT_NAME="${TARGET_BIN_OUTPUT_NAME}"
)
