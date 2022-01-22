#pragma once

#include "Image.h"

void xblur(Image &b, Image const &a, int nblur);
void yblur(Image &b, Image const &a, int nblur);
void boxblur(Image &a, int nxblur, int nyblur);
