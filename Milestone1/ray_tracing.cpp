#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <time.h>

using namespace std;

const double PI = 3.1415926;

double vec3Norm(double, double, double);
double vec3DotProd(double[3], double[3]);
double square(double);
void uniformSphereSampling(double[3]);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
void getCoordFromWvec(int coord[2], int dimN, double W[3], double Wmax);
void serialRayTracing(double **G, int dimN, double C[3], double R, double L[3], double Wy, double Wmax, int nRays);

int main(int argc, char *argv[])
{
    int nRays = stoi(argv[1]);
    int dimN = stoi(argv[2]);
    double **G = dmatrix(0, dimN - 1, 0, dimN - 1);
    double C[3] = {0, 12, 0};
    double R = 6;
    double L[3] = {4, 4, -1};
    double Wy = 10;
    double Wmax = 10;

    std::clock_t c_start = std::clock();

    serialRayTracing(G, dimN, C, R, L, Wy, Wmax, nRays);

    std::clock_t c_end = std::clock();

    long double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";

    fstream out;
    out.open(std::to_string(nRays) + ".dat", fstream::out | fstream::trunc);
    // out.precision(6);
    for (int k = 0; k < dimN; k++)
    {
        for (int l = 0; l < dimN; l++)
        {
            out << std::scientific << G[k][l] << " ";
        }
        out << endl;
    }
    out.close();

    return 0;
}
/**
 * @brief Implementation of Algorithm 2 in a serial way (singly threaded) on a CPU
 *
 * @param G Window Grids
 * @param C Sphere center
 * @param R Sphere radius
 * @param L Lightsource coord
 * @param Wy Window's y-coord
 * @param Wmax Window's size on the x,z-plane, Wmax>0
 * @param dimN Window Grid dimension
 * @param nRays Number of rays to be sampled
 */
void serialRayTracing(double **G, int dimN, double C[3], double R, double L[3], double Wy, double Wmax, int nRays)
{
    for (int i = 0; i < dimN; i++)
    {
        for (int j = 0; j < dimN; j++)
        {
            G[i][j] = 0;
        }
    }
    double V[3] = {0, 0, 0};
    double W[3] = {0, Wy, 0};
    double t, val, b;
    double I[3];
    double N[3];
    double S[3];
    int coord[2];

    for (int n = 0; n < nRays; n++)
    {
        while (abs(W[0]) >= Wmax || abs(W[2]) >= Wmax || square(vec3DotProd(V, C)) + square(R) <= vec3DotProd(C, C))
        {
            uniformSphereSampling(V);

            W[0] = Wy * V[0] / V[1];
            W[2] = Wy * V[2] / V[1];
        }

        t = vec3DotProd(V, C) - sqrt(square(vec3DotProd(V, C)) + square(R) - vec3DotProd(C, C));
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

        getCoordFromWvec(coord, dimN, W, Wmax);
        G[coord[0]][coord[1]] += b;
        V[0] = 0;
        V[1] = 0;
        V[2] = 0;
    }
}

void getCoordFromWvec(int coord[2], int dimN, double W[3], double Wmax)
{
    double gridSize = (double)2 * Wmax / (double)dimN;
    coord[0] = (int)floor((W[0] + Wmax) / gridSize); // Wx
    coord[1] = (int)floor((W[2] + Wmax) / gridSize); // Wz
}

void uniformSphereSampling(double V[3])
{
    double phi = PI * (double)rand() / (double)RAND_MAX;
    double cosTheta = (double)2 * (double)rand() / (double)RAND_MAX - 1;
    double sinTheta = sqrt(1 - square(cosTheta));
    V[0] = sinTheta * cos(phi);
    V[1] = sinTheta * sin(phi);
    V[2] = cosTheta;
}

double vec3Norm(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

double vec3DotProd(double A[3], double B[3])
{
    return (A[0] * B[0] + A[1] * B[1] + A[2] * B[2]);
}

double square(double x)
{
    return x * x;
}

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch]                          \
 */
{
    long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
    double **m;
    double x, y;
    /* allocate pointers to rows */
    m = (double **)malloc((size_t)((nrow + 1) * sizeof(double *)));
    m += 1;
    m -= nrl;
    /* allocate rows and set pointers to them */
    m[nrl] = (double *)malloc((size_t)((nrow * ncol + 1) * sizeof(double)));
    m[nrl] += 1;
    m[nrl] -= ncl;
    for (i = nrl + 1; i <= nrh; i++)
        m[i] = m[i - 1] + ncol;
    /* return pointer to array of pointers to rows */
    return m;
}
