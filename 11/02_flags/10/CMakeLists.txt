add_executable(main)
file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h)
target_sources(main PUBLIC ${sources})

target_compile_definitions(main PUBLIC MY_NAME="The ${CMAKE_CXX_COMPILER_ID} Compiler")
