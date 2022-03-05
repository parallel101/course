add_executable(main)
file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h)
target_sources(main PUBLIC ${sources})

target_compile_definitions(main PUBLIC
    $<$<PLATFORM_ID:Windows>:MY_NAME="Bill Gates">
    $<$<PLATFORM_ID:Linux>:MY_NAME="Linus Torvalds">
    $<$<PLATFORM_ID:Darwin>:MY_NAME="Steve Jobs">
    )
