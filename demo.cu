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
#include "zkprod.cuh"

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

    uint c_size_count = 0, p_size_count = 0;
    Timer c_timer, p_timer, v_timer;
    c_timer.start();
    zkMatMul matmul(A, B, num, m, n, k, genA, genB, p_timer, v_timer, c_size_count);
    c_timer.stop();
    matmul.prove(genA, genB, p_size_count);

    // Testing zkReLU
    uint relu_dim = stoi(argv[5]);
    auto Z = FrTensor::random_int(num * relu_dim, 32);
    auto GA = FrTensor::random_int(num * relu_dim, 32);

    auto gen_relu = Commitment::random_generators(1 << (ceilLog2(num * relu_dim)/2 + 4));
    c_timer.start();
    zkReLU relu(Z, GA, gen_relu, p_timer, v_timer, c_size_count);
    c_timer.stop();
    relu.prove(gen_relu, p_size_count);

    // Testing zkProd
    uint prod_dim = stoi(argv[6]);
    auto A1 = FrTensor::random_int(num * prod_dim, 32);
    auto A2 = FrTensor::random_int(num * prod_dim, 32);
    auto prod_gen = Commitment::random_generators(1 << (ceilLog2(num * prod_dim)/2 + 1));
    c_timer.start();
    zkProd prod(A1, A2, num, prod_dim, prod_gen, prod_gen, p_timer, v_timer, c_size_count);
    c_timer.stop();
    prod.prove(prod_gen, prod_gen, p_size_count);
    // zkProd(const FrTensor& A, const FrTensor& B, uint num, uint dim, Commitment& genA, Commitment& genB, Timer& p_timer, Timer& v_timer, uint& commit_size_count)

    cout << "Commitment size: " << c_size_count << " bytes." << endl;
    cout << "Proof size: " << p_size_count << " bytes." << endl;
    cout << "Committing time: " << c_timer.getTotalTime() << " seconds." << endl;
    cout << "Prover time: " << p_timer.getTotalTime() << " seconds." << endl;
    cout << "Verifier time: " << v_timer.getTotalTime() << " seconds." << endl;
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
}