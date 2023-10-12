#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
// #include <torch/torch.h>
// #include <torch/script.h>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "commitment.cuh"
#include "proof.cuh"

#include "timer.hpp"
#include "zkfc.cuh"
#include "zkrelu.cuh"
#include "zkmatmul.cuh"

using namespace std;

int main(int argc, char *argv[])
{   
    // Testing zkMatMul
    uint num = stoi(argv[1]);
    uint m = stoi(argv[2]);
    uint n = stoi(argv[3]);
    uint k = stoi(argv[4]);

    auto A = FrTensor::random_int(num * m * n, 32);
    auto B = FrTensor::random_int(num * n * k, 32);

    zkMatMul matmul(A, B, num, m, n, k);
    matmul.prove(random_vec(ceilLog2(num)), random_vec(ceilLog2(num)), random_vec(ceilLog2(m)), random_vec(ceilLog2(n)), random_vec(ceilLog2(k)));


    // uint relu_size = stoi(argv[1]);
    // auto Z = FrTensor::random_int(relu_size, 32);
    // auto GA = FrTensor::random_int(relu_size, 32);
    // cout << Z << endl;
    // cout << GA << endl;
    // zkReLU relu(Z, GA);
    // cout << relu.A << endl;
    // cout << relu.GZ << endl;

    // cout << relu.aux << endl;
}