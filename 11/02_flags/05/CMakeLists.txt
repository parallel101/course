add_executable(main)
file(GLOB sources CONFIGURE_DEPENDS *.cpp *.h)
target_sources(main PUBLIC ${sources})

target_compile_definitions(main PUBLIC
    $<$<PLATFORM_ID:Windows>:MY_NAME="DOS-like">
    $<$<PLATFORM_ID:Linux,Darwin,FreeBSD>:MY_NAME="Unix-like">
    )
