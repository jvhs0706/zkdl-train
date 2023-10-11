#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <torch/torch.h>
#include <torch/script.h>
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

using namespace std;

int main(int argc, char *argv[])
{
    uint relu_size = stoi(argv[1]);
    auto Z = FrTensor::random_int(relu_size, 32);
    auto GA = FrTensor::random_int(relu_size, 32);
    cout << Z << endl;
    cout << GA << endl;
    zkReLU relu(Z, GA);

}