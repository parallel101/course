add_executable(main)
file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h)
target_sources(main PUBLIC ${sources})

target_compile_definitions(main PUBLIC
    $<$<CXX_COMPILER_ID:GNU,Clang>:MY_NAME="Open-source">
    $<$<CXX_COMPILER_ID:MSVC,NVIDIA>:MY_NAME="Commercial">
    )
