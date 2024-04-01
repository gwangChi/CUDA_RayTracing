
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <fstream>

__device__ __constant__ float PI = 3.1415926;

void list_env_properties();

__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__device__ float getRand(curandState *globalState, int id) {
    curandState localState = globalState[id];
    float RANDOM = curand_uniform(&localState);
    globalState[id] = localState;
    return RANDOM;
}
__device__ void getCoordFromWvec(int coord[2], int dimN, float W[3], float Wmax);
__device__ float square(const float&& x);
__device__ float vec3DotProd(float A[3], float B[3]);
__device__ float vec3Norm(const float&& x, const float&& y, const float&& z);
__global__ void threadRayTracing(curandState *globalState, float* dev_G, int dimG, float Cx, float Cy, float Cz, float R, float Lx, float Ly, float Lz,
    float Wy, float Wmax, int nRays);
__device__ void uniformSphereSampling(float V[3], curandState* globalState, int id);

int main(int argc, char** argv)
{
    list_env_properties();

    cudaEvent_t start_device, stop_device; /* CUDA timers */
    //cudaStream_t computeStream; /* streams thread creation */
    curandState* devStates;
    float time_device; /* timer */

    int nRays = std::stoi(argv[1]);
    int dimG = std::stoi(argv[2]);

    float *G, *dev_G;     /* Window Grid contains shading values */
    float C[3] = { 0, 12, 0 };
    float R = 6;
    float L[3] = { 4, 4, -1 };
    float Wy = 10;
    float Wmax = 10;

    G = (float*)malloc(dimG * dimG * sizeof(float));

    for (size_t i = 0; i < dimG * dimG; i++) {
        G[i] = 0.;
    }

    /***************************************************************************************/
    const int RAYS_PER_THREAD = 10000;
    const int THREADS_PER_BLOCK = 256;
    /****************************************************************************************/

    int NUM_BLOCKS = (nRays % (THREADS_PER_BLOCK * RAYS_PER_THREAD) == 0) ?
        nRays / (THREADS_PER_BLOCK * RAYS_PER_THREAD) : (nRays / (THREADS_PER_BLOCK * RAYS_PER_THREAD) + 1);

    cudaMalloc((void**)&devStates, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState));
    srand(time(0));
    int seed = rand();
    setup_kernel <<<NUM_BLOCKS, THREADS_PER_BLOCK >>> (devStates, seed);

    cudaMalloc((void**) &dev_G, dimG * dimG * sizeof(float));
    cudaMemcpy(dev_G, G, dimG * dimG * sizeof(float), cudaMemcpyHostToDevice);



    /* Creates timers but not start yet */
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);
    //cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking);

    cudaEventRecord(start_device, 0);   /* starts recording time */

    //threadRayTracing<<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, computeStream>>>(time(NULL), dev_G, dimG, C[0], C[1], C[2], R, L[0], L[1], L[2], Wy, Wmax, RAYS_PER_THREAD);

    threadRayTracing <<<NUM_BLOCKS, THREADS_PER_BLOCK >>> (devStates, dev_G, dimG, C[0], C[1], C[2], R, L[0], L[1], L[2], Wy, Wmax, RAYS_PER_THREAD);

    cudaEventRecord(stop_device, 0);
    cudaEventSynchronize(stop_device);
    cudaEventElapsedTime(&time_device, start_device, stop_device);  /* stops recording time*/
    printf("Ended! Time used: %f(s)\n", time_device/1000.);

    cudaMemcpy(G, dev_G, dimG * dimG * sizeof(float), cudaMemcpyDeviceToHost);    /* Copies global G to host G */
    cudaFree(dev_G);


    std::fstream out;
    out.open(std::to_string(nRays) + ".dat", std::fstream::out | std::fstream::trunc);
    // out.precision(6);
    for (int k = 0; k < dimG; k++)
    {
        for (int l = 0; l < dimG; l++)
        {
            out << std::scientific << G[k*dimG + l] << " ";
        }
        out << std::endl;
    }
    out.close();
    free(G);

    return 0;
}
/**
 * @brief Implementation of Algorithm 2 on GPU
 *
 * @param G Window Grids
 * @param C Sphere center
 * @param R Sphere radius
 * @param L Lightsource coord
 * @param Wy Window's y-coord
 * @param Wmax Window's size on the x,z-plane, Wmax>0
 * @param dimG Window Grid's dimension
 * @param nRays Number of rays to be sampled
 */
__global__ void threadRayTracing(curandState *globalState, float* dev_G, int dimG, float Cx, float Cy, float Cz, float R, float Lx, float Ly, float Lz, 
                                 float Wy, float Wmax, int nRays) {
    float C[3] = { Cx, Cy, Cz };
    float L[3] = { Lx, Ly, Lz};
    float V[3] = { 0, 0, 0 };
    float W[3] = { 0, Wy, 0 };
    float t, val, b;
    float I[3];
    float N[3];
    float S[3];
    int coord[2];
    int randCallCount = 0;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = 0; n < nRays; n++)
    {
        while (abs(W[0]) >= Wmax || abs(W[2]) >= Wmax || square(vec3DotProd(V, C)) + square(std::move(R)) <= vec3DotProd(C, C))
        {
            uniformSphereSampling(V, globalState, id);

            W[0] = Wy * V[0] / V[1];
            W[2] = Wy * V[2] / V[1];

            randCallCount++;
        }

        //printf("Thread %d from block %d gets V[ %f, %f, %f] in %d rand calls.\n", threadIdx.x, blockIdx.x, V[0], V[1], V[2], randCallCount);
        randCallCount = 0;

        t = vec3DotProd(V, C) - sqrt(square(vec3DotProd(V, C)) + square(std::move(R)) - vec3DotProd(C, C));
        I[0] = t * V[0];
        I[1] = t * V[1];
        I[2] = t * V[2];

        val = vec3Norm(I[0] - C[0], I[1] - C[1], I[2] - C[2]);

        N[0] = (I[0] - C[0]) / val;
        N[1] = (I[1] - C[1]) / val;
        N[2] = (I[2] - C[2]) / val;

        val = vec3Norm(L[0] - I[0], L[1] - I[1], L[2] - I[2]);

        S[0] = (L[0] - I[0]) / val;
        S[1] = (L[1] - I[1]) / val;
        S[2] = (L[2] - I[2]) / val;

        val = vec3DotProd(S, N);
        b = (val >= 0) ? val : 0;

        getCoordFromWvec(coord, dimG, W, Wmax);
        atomicAdd(& dev_G[coord[0] * dimG + coord[1]], b);

        //printf("G[%d,%d] is now %f after adding %f\n", coord[0], coord[1], dev_G[coord[0] * dimG + coord[1]], b);

        V[0] = 0;
        V[1] = 0;
        V[2] = 0;
    }
}

__device__ void getCoordFromWvec(int coord[2], int dimN, float W[3], float Wmax)
{
    float gridSize = (float)2 * Wmax / (float)dimN;
    coord[0] = (int)floor((W[0] + Wmax) / gridSize); // Wx
    coord[1] = (int)floor((W[2] + Wmax) / gridSize); // Wz
}

__device__ void uniformSphereSampling(float V[3], curandState *globalState, int id)
{
    float phi = PI * getRand(globalState, id);
    float cosTheta = 2. * getRand(globalState, id) - 1.;
    float sinTheta = sqrt(1. - square(std::move(cosTheta)));
    V[0] = sinTheta * cos(phi);
    V[1] = sinTheta * sin(phi);
    V[2] = cosTheta;
}

__device__ float vec3Norm(const float&& x, const float&& y, const float&& z)
{
    return sqrt(x * x + y * y + z * z);
}

__device__ float vec3DotProd(float A[3], float B[3])
{
    return (A[0] * B[0] + A[1] * B[1] + A[2] * B[2]);
}

__device__ float square(const float&& x)
{
    return x * x;
}

void list_env_properties() {
    int deviceCount, device;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
            if (device == 0)
            {
                printf("name:%s\n", properties.name);
                printf("memory:%lu\n", properties.totalGlobalMem);
                printf("warpsize:%d\n", properties.warpSize);
                printf("max threads per block:%d\n", properties.maxThreadsPerBlock);
                printf("clock rate:%d\n", properties.clockRate);
                printf("multiProcessorCount %d\n", properties.multiProcessorCount);
                printf("maxThreadsPerMultiProcessor %d\n", properties.maxThreadsPerMultiProcessor);
            }
    }
}
