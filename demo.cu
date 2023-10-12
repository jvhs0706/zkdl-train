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

    auto A = FrTensor::random_int(num * m * n, 32).mont();
    auto B = FrTensor::random_int(num * n * k, 32).mont();

    auto genA = Commitment::random_generators(1 << (ceilLog2(num * m * n)/2 + 1));
    auto genB = Commitment::random_generators(1 << (ceilLog2(num * n * k)/2 + 1));

    Timer timer;
    timer.start();
    zkMatMul matmul(A, B, num, m, n, k, genA, genB);
    timer.stop();
    cout << "Commit time: " << timer.getTotalTime() << " seconds." << endl;
    timer.reset();
    timer.start(); 
    matmul.prove(genA, genB);
    timer.stop();
    cout << "Proof time: " << timer.getTotalTime() << " seconds." << endl;
    timer.reset();
    cout << "Current CUDA status: " << cudaGetLastError() << endl;


    // Testing zkReLU
    uint relu_dim = stoi(argv[5]);
    auto Z = FrTensor::random_int(relu_dim, 32);
    auto GA = FrTensor::random_int(relu_dim, 32);

    auto gen_relu = Commitment::random_generators(1 << (ceilLog2(relu_dim)/2 + 4));
    timer.start();
    zkReLU relu(Z, GA, gen_relu);
    timer.stop();
    cout << "zkReLU commit time: " << timer.getTotalTime() << " seconds." << endl;
    timer.reset();
    timer.start();
    relu.prove(gen_relu);
    timer.stop();
    cout << "zkReLU proof time: " << timer.getTotalTime() << " seconds." << endl;
}