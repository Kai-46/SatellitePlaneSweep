file(GLOB PSL_BASE_SRC_FILES
        "psl_base/*.cpp"
        "psl_base/*.h"
        )

file(GLOB PSL_CUDA_BASE_SRC_FILES
        "psl_stereo/*.cu"
        "psl_stereo/*.cuh"
        )

file(GLOB PSL_STEREO_SRC_FILES
        "psl_stereo/*.cpp"
        "psl_stereo/*.h"
        )

cuda_compile(PSL_CUDA_SRC_FILES
        psl_cudaBase/deviceBuffer.cuh
        psl_cudaBase/deviceBuffer.cu
        psl_cudaBase/deviceImage.cuh
        psl_cudaBase/cudaBackbone.cu
        OPTIONS
        -use_fast_math
        #-arch sm_10
        #-Xcompiler -fpic
        )

add_library(PSL STATIC
        ${PSL_BASE_SRC_FILES}
        ${PSL_STEREO_SRC_FILES}
        ${PSL_CUDA_BASE_SRC_FILES}
        ${PSL_CUDA_SRC_FILES}
        )

target_link_libraries(PSL
        ${OpenCV_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        )
