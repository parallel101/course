add_executable(main)
file(GLOB sources *.cpp *.h)
target_sources(main PUBLIC ${sources})
