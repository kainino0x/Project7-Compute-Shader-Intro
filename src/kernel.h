#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace Nbody {
void step(float dt);
void init(int N);
void updatePBO(float4 * pbodptr, int width, int height);
void updateVBO(float * vbodptr, int width, int height);
}
