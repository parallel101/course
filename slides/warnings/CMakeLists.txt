cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_COMPILER clang++)

project(main)

add_compile_options(
    -Werror
    -Wconversion
    -Wunused
    -Wall
    -Wextra
    -Wuninitialized
    -Weffc++
    -Wno-non-virtual-dtor
    -Wdelete-non-virtual-dtor
    -Wzero-as-null-pointer-constant
    -Wparentheses
    -Wempty-body
    -Wsign-compare
    -Wsign-conversion
    -Wold-style-cast
    -Wrange-loop-construct
    -Wformat
    -Wstrict-aliasing=3
    -Wrestrict
    -Winvalid-memory-model
    -Wsuggest-override
    -Wattributes
    -Wmissing-declarations
    -Warray-bounds
    -Wdangling-pointer=2
    -Wreturn-local-addr
    -Wuse-after-free
    -Wmismatched-new-delete
    -Wvla
    -Walloca
    -Wreorder
    -Woverflow
    -Werror=return-type
    -Wswitch
    -Wimplicit-fallthrough
    # -Wshadow=local
# -Werror
# -Wall
# -Wextra
# -Weffc++
# -Wunused
# -Wunused-but-set-variable
# -Wunused-but-set-parameter
# -Wunused-const-variable
# -Wunused-value
# -Werror=unused-result
# -Werror=use-after-free=3
# -Werror=string-compare
# -Werror=return-type
# -Werror=deprecated-copy-dtor
# -Werror=deprecated-copy
# -Wnon-virtual-dtor
# -Werror=delete-non-virtual-dtor
# -Werror=mismatched-new-delete
# -Werror=suggest-override
# -Werror=conversion-null
# -Werror=zero-as-null-pointer-constant
# -Werror=reorder
# -Werror=class-memaccess
# -Wparentheses
# -Wconversion
# -Werror=float-conversion
# -Wenum-conversion
# -Wsign-conversion
# -Wsign-compare
# -Wsign-promo
# -Wold-style-cast
# -Wrange-loop-construct
# -Werror=uninitialized
# -Werror=maybe-uninitialized
# -Wmissing-field-initializers
# -Wmissing-declarations
# -Wempty-body
# -Wlogical-op
# -Wlogical-not-parentheses
# -Werror=address
# -Werror=multichar
# -Werror=overflow
# -Werror=restrict
# -Werror=strict-aliasing
# -Werror=invalid-memory-model
# -Werror=vla
# -Wattributes
# -Wstrict-overflow
# -Werror=stringop-overflow
# -Werror=shift-overflow
)

file(GLOB sources "*.cpp")
add_executable(main ${sources})
