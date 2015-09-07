CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

Instructions (delete me)
========================

This is due Monday, September 7.

**Summary:** In this project, you will get some real experience writing simple
CUDA kernels, using them, and analyzing their performance. You'll implement the
simulation step of an N-body simulation, and you'll write some GPU-accelerated
matrix math operations.

## Part 0: Nothing New

This project (and all other CUDA projects in this course) requires an NVIDIA
graphics card with CUDA capability. Any card with Compute Capability 2.0
(`sm_20`) or greater will work. Check your GPU on this
[compatibility table](https://developer.nvidia.com/cuda-gpus).
If you do not have a personal machine with these specs, you may use those
computers in the Moore 100B/C which have supported GPUs.

**HOWEVER**: If you need to use the lab computer for your development, you will
not presently be able to do GPU performance profiling. This will be very
important for debugging performance bottlenecks in your program. If you do not
have administrative access to any CUDA-capable machine, please email the TA.

## Part 1: N-body Simulation

### 1.0. The Usual

See Project 0, Parts 1-3 for reference.

If you are using the Nsight IDE (not Visual Studio) and started Project 0
early, note that things have
changed slightly. Instead of creating a new project, use
*File->Import->General->Existing Projects Into Workspace*, and select the
`Project1-Part1` folder as the root directory. Under *Project->Build
Configurations->Set Active...*, you can now select various Release and Debug
builds.

* `src/` contains the source code.
* `external/` contains the binaries and headers for GLEW, GLFW, and GLM.

**CMake note:** Do not change any build settings or add any files to your
project directly (in Visual Studio, Nsight, etc.) Instead, edit the
`src/CMakeLists.txt` file. Any files you create must be added here. If you edit
it, just rebuild your VS/Nsight project to sync the changes into the IDE.


### 1.1. CUDA Done That With My Eyes Closed

To get used to using CUDA kernels, you'll write simple CUDA kernels and
kernel invocations for performing an N-body gravitational simulation.
The following source files are included in the project:

* `src/main.cpp`: Performs all of the CUDA/OpenGL setup and OpenGL
  visualization.
* `src/kernel.cu`: CUDA device functions, state, kernels, and CPU functions for
  kernel invocations.

1. Search the code for `TODO`:
   * `src/kernel.cu`: Use what you learned in the first lectures to
     figure out how to resolve these 4 TODOs.

Take a screenshot. Commit and push your code changes.


## Part 2: Matrix Math

In this part, you'll set up a CUDA project with some simple matrix math
functionality. Put this in the `Project1-Part2` directory in your repository.

### 1.1. Create Your Project

You'll need to copy over all of the boilerplate project-related files from
Part 1:

* `cmake/`
* `external/`
* `.cproject`
* `.project`
* `GNUmakefile`
* `CMakeLists.txt`
* `src/CMakeLists.txt`

Next, create empty text files for your main function and CUDA kernels:

* `src/main.cpp`
* `src/matrix_math.h`
* `src/matrix_math.cu`

As you work through the next steps, find and use relevant code from Part 1 to
get the new project set up: includes, error checking, initialization, etc.

### 1.2. Setting Up CUDA Memory

As discussed in class, there are two separate memory spaces: host memory and
device memory. Host memory is accessible by the CPU, while device memory is
accessible by the GPU.

In order to allocate memory on the GPU, we need to use the CUDA library
function `cudaMalloc`. This reserves a portion of the GPU memory and returns a
pointer, like standard `malloc` - but the pointer returned by `cudaMalloc` is
in the GPU memory space and is only accessible from GPU code. You can use
`cudaFree` to free GPU memory allocated using `cudaMalloc`.

We can copy memory to and from the GPU using `cudaMemcpy`. Like C `memcpy`,
you will need to specify the size of memory that you are copying. But
`cudaMemcpy` has an additional argument - the last argument specifies the
whether the copy is from host to device, device to host, device to device, or
host to host.

* Look up documentation on `cudaMalloc`, 'cudaFree', and `cudaMemcpy` to find
  out how to use them - they're not quite obvious.

In an initialization function in `matrix_math.cu`, initialize three 5x5 matrices
on the host and three on the device. Prefix your variables with `hst_` and
`dev_`, respectively, so you know what kind of pointers they are!
These arrays can each be represented as a 1D array of floats:

`{ A_00, A_01, A_02, A_03, A_04, A_10, A_11, A_12, ... }`

You should also create cleanup method(s) to free the CPU and GPU memory you
allocated. Don't forget to initialize and cleanup in main!

### 1.3. Creating CUDA Kernels

Given 5x5 matrices A, B, and C (each represented as above), implement the
following functions as CUDA kernels (`__global__`):

* `mat_add(A, B, C)`: `C` is overwritten with the result of `A + B`
* `mat_sub(A, B, C)`: `C` is overwritten with the result of `A - B`
* `mat_mul(A, B, C)`: `C` is overwritten with the result of `A * B`

You should write some tests to make sure that the results of these operations
are as you expect.

Tips:

* `__global__` and `__device__` functions only have access to memory that is
  stored on the device. Any data that you want to use on the CPU or GPU must
  exist in the right memory space. If you need to move data, you can use
  `cudaMemcpy`.
* The triple angle brackets `<<< >>>` provide parameters to the CUDA kernel
  invocation: `<<<blocks_per_tile, threads_per_block, ...>>>`.
* Don't worry if your IDE doesn't understand some CUDA syntax (e.g.
  `__device__` or `<<< >>>`). By default, it may not understand CUDA
  extensions.


## Part 3: Performance Analysis

For this project, we will guide you through your performance analysis with some
basic questions. In the future, you will guide your own performance analysis -
but these simple questions will always be critical to answer. In general, we
want you to go above and beyond the suggested performance investigations and
explore how different aspects of your code impact performance as a whole.

The provided framerate meter (in the window title) will be a useful base
metric, but adding your own `cudaTimer`s, etc., will allow you to do more
fine-grained benchmarking of various parts of your code.

REMEMBER:
* Performance should always be measured relative to some baseline when
  possible. A GPU can make your program faster - but by how much?
* If a change impacts performance, show a comparison. Describe your changes.
* Describe the methodology you are using to benchmark.
* Performance plots are a good thing.

### Questions

For Part 1, there are two ways to measure performance:
* Disable visualization so that the framerate reported will be for the the
  simulation only, and not be limited to 60 fps. This way, the framerate
  reported in the window title will be useful.
  * Change `#define VISUALIZE` to `0`.
* For tighter timing measurement, you can use CUDA events to measure just the
  simulation CUDA kernel. Info on this can be found online easily. You will
  probably have to average over several simulation steps, similar to the way
  FPS is currently calculated.

**Answer these:**

* Parts 1 & 2: How does changing the tile and block sizes affect performance?
  Why?
* Part 1: How does changing the number of planets affect performance? Why?
* Part 2: Without running comparisons of CPU code vs. GPU code, how would you
  expect the performance to compare? Why? What might be the trade-offs?

**NOTE: Nsight performance analysis tools *cannot* presently be used on the lab
computers, as they require administrative access.** If you do not have access
to a CUDA-capable computer, the lab computers still allow you to do timing
mesasurements! However, the tools are very useful for performance debugging.


## Part 4: Write-up

1. Update all of the TODOs at the top of this README.
2. Add your performance analysis.


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project 0: PENNKEY`
   * Direct link to your pull request on GitHub
   * In the form of a grade (0-100+), evaluate your own performance on the
     project.
   * Feedback on the project itself, if any.

And you're done!
