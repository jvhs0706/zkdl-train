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
    bool seq = (argc > 3);
    uint depth = stoi(argv[1]);
    uint num = seq ? 1 : (16 * depth);
    uint batch_size = 64;
    uint width = stoi(argv[2]);
    

    uint c_size_count = 0, p_size_count = 0;
    Timer c_timer, p_timer, v_timer;

    auto A = FrTensor::random_int(num * batch_size * width, 32).mont();
    auto W = FrTensor::random_int(num * width * width, 32).mont();
    auto GZ = FrTensor::random_int(num * batch_size * width, 32).mont();

    auto genA = Commitment::random_generators(1 << (ceilLog2(A.size)/2 + 1));
    auto genW = Commitment::random_generators(1 << (ceilLog2(W.size)/2 + 1));

    c_timer.start();
    zkMatMul fc_Z(A, W, num, batch_size, width, width, genA, genW, p_timer, v_timer, c_size_count);
    zkMatMul fc_GW(GZ, A, num, width, batch_size, width, genA, genA, p_timer, v_timer, c_size_count);
    zkMatMul fc_GA(GZ, W, num, batch_size, width, width, genA, genW, p_timer, v_timer, c_size_count);
    c_timer.stop();

    fc_Z.prove(genA, genW, p_size_count);
    fc_GW.prove(genA, genA, p_size_count);
    fc_GA.prove(genA, genW, p_size_count);

    uint relu_dim = batch_size * width;
    auto Z = FrTensor::random_int(num * relu_dim, 32);
    auto GA = FrTensor::random_int(num * relu_dim, 32);
    auto gen_relu = Commitment::random_generators(1 << (ceilLog2(num * relu_dim)/2 + 4));
    c_timer.start();
    zkReLU relu(Z, GA, gen_relu, p_timer, v_timer, c_size_count);
    c_timer.stop();
    relu.prove(gen_relu, p_size_count);

    // Testing zkProd
    uint prod_dim = batch_size * width * 2;
    auto A1 = FrTensor::random_int(num * prod_dim * 2, 32);
    auto A2 = FrTensor::random_int(num * prod_dim * 2, 32);
    auto prod_gen = Commitment::random_generators(1 << (ceilLog2(num * prod_dim)/2 + 1));
    c_timer.start();
    zkProd prod(A1, A2, num, prod_dim * 2, prod_gen, prod_gen, p_timer, v_timer, c_size_count);
    c_timer.stop();
    prod.prove(prod_gen, prod_gen, p_size_count);

    if (seq){
        cout << "seq," << depth <<","<< width << "," << c_size_count*depth << "," << p_size_count*depth << "," << c_timer.getTotalTime()*depth << "," << p_timer.getTotalTime()*depth << "," << v_timer.getTotalTime()*depth << endl;
    }
    else {
        cout << "fac4dnn," <<depth <<","<< width << "," << static_cast<float>(c_size_count)/16 << "," << static_cast<float>(p_size_count)/16 << "," << c_timer.getTotalTime()/16 << "," << p_timer.getTotalTime()/16 << "," << v_timer.getTotalTime()/16 << endl;
    }
    

    // cout << "Commitment size: " << c_size_count << " bytes." << endl;
    // cout << "Proof size: " << p_size_count << " bytes." << endl;
    // cout << "Committing time: " << c_timer.getTotalTime() << " seconds." << endl;
    // cout << "Prover time: " << p_timer.getTotalTime() << " seconds." << endl;
    // cout << "Verifier time: " << v_timer.getTotalTime() << " seconds." << endl;
    // cout << "Current CUDA status: " << cudaGetLastError() << endl;
}