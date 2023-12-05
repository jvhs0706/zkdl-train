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
    uint c_size_count = 0, p_size_count = 0;
    Timer c_timer, p_timer, v_timer;
    
    // define your own model here

    cout << "Commitment size: " << c_size_count << " bytes." << endl;
    cout << "Proof size: " << p_size_count << " bytes." << endl;
    cout << "Committing time: " << c_timer.getTotalTime() << " seconds." << endl;
    cout << "Prover time: " << p_timer.getTotalTime() << " seconds." << endl;
    cout << "Verifier time: " << v_timer.getTotalTime() << " seconds." << endl;
    cout << "Current CUDA status: " << cudaGetLastError() << endl;
}