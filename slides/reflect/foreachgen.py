# this python scripts generates:
#define REFLECT__PP_FOREACH_1(f, _1) f(_1)
#define REFLECT__PP_FOREACH_2(f, _1, _2) f(_1) f(_2)
#define REFLECT__PP_FOREACH_3(f, _1, _2, _3) f(_1) f(_2) f(_3)
#define REFLECT__PP_FOREACH_4(f, _1, _2, _3, _4) f(_1) f(_2) f(_3) f(_4)
#define REFLECT__PP_FOREACH_5(f, _1, _2, _3, _4, _5) f(_1) f(_2) f(_3) f(_4) f(_5)
# ...
#define REFLECT__PP_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, N, ...) N
#define REFLECT__PP_NARGS(...) REFLECT__PP_NARGS_IMPL(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

import sys

n = int(sys.argv[1])
for i in range(1, n+1):
    print("#define REFLECT__PP_FOREACH_{0}(f, {1}) ".format(i, ", ".join(["_"+str(j) for j in range(1, i+1)])), end="")
    print(" ".join(["f(_{0})".format(j) for j in range(1, i+1)]))

print("#define REFLECT__PP_NARGS_IMPL({0}, N, ...) N".format(", ".join(["_"+str(j) for j in range(1, n+1)])))
print("#define REFLECT__PP_NARGS(...) REFLECT__PP_NARGS_IMPL(__VA_ARGS__, {0})".format(", ".join([str(j) for j in range(n, 0, -1)])))
