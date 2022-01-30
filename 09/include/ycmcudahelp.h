#pragma once

#include <cuda_runtime.h>

#ifdef __ycm_cuda__

template <class T>
void surf2Dwrite(T, cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap);

template <class T>
T surf2Dread(cudaSurfaceObject_t, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap);

template <class T>
void surf3Dwrite(T, cudaSurfaceObject_t, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap);

template <class T>
T surf3Dread(cudaSurfaceObject_t surf, int, int, int, cudaSurfaceBoundaryMode = cudaBoundaryModeTrap);

#endif
