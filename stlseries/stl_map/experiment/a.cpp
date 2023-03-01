#include "preprocess.h"

#define FUN(x) hello-x

int arr[] = {PP_FOREACH(FUN, PP_COMMA(), 1, 2, 3)};
