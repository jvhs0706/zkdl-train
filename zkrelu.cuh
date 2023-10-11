#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 

const uint zkReLU_Q = 32;
const uint zkReLU_R = 16;
const uint zkReLU_B = zkReLU_Q + zkReLU_R;

class zkReLU {
protected:
    const uint size;
    FrTensor Z;
    FrTensor GA;

    FrTensor A;
    FrTensor GZ;
    
    FrTensor aux;
public:
    zkReLU(const FrTensor& Z, const FrTensor& GA);
};

DEVICE long scalar_to_long(Fr_t num){
    if (num.val[7] == 1944954707U) {
        Fr_t abs = blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, num);
        return -(static_cast<long>(abs.val[0]) | (static_cast<long>(abs.val[1]) << 32) );
    }
    else if (num.val[7] == 0U) return static_cast<long>(num.val[0]) | (static_cast<long>(num.val[1]) << 32);
    else return 0;
}

DEVICE Fr_t long_to_scalar(long num){
    if (num < 0) {
        long abs = -num;
        return blstrs__scalar__Scalar_sub({0, 0, 0, 0, 0, 0, 0, 0}, {static_cast<uint>(abs), static_cast<uint>(abs >> 32), 0, 0, 0, 0, 0, 0});
    }
    else return {static_cast<uint>(num), static_cast<uint>(num >> 32), 0, 0, 0, 0, 0, 0};
}

DEVICE long rescale(long num){
    num += (1L << (zkReLU_R - 1));
    num >>= zkReLU_R;
    return num;
}

// Broadcast montegomery multiplication
KERNEL void zkReLU_init_kernel(GLOBAL Fr_t* Z_ptr, GLOBAL Fr_t* GA_ptr, GLOBAL Fr_t* A_ptr, GLOBAL Fr_t* GZ_ptr, GLOBAL Fr_t* aux_ptr, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;
    long z = scalar_to_long(Z_ptr[gid]);
    long ga = scalar_to_long(GA_ptr[gid]);
    bool mask = z >= 0;
    A_ptr[gid] = long_to_scalar(rescale(mask * z));
    GZ_ptr[gid] = long_to_scalar(rescale(mask * ga)); 
    #pragma unroll
    for (uint i = 0; i < zkReLU_B; ++ i) {
        aux_ptr[gid * zkReLU_B + i] = ((z >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
        aux_ptr[(gid + n) * zkReLU_B + i] = ((ga >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
    }
}

zkReLU(const FrTensor& Z, const FrTensor& GA): size(Z.size), Z(Z), GA(GA), GZ(Z.size), A(GA.size), aux(2 * Z.size * (zkReLU_Q + zkReLU_R))
{
    // make sure the four inputs are of the same size
    if (GA.size != size) throw std::invalid_argument("Z and GA must be of the same size");
    zkReLU_init_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(Z.gpu_data, GA.gpu_data, A.gpu_data, GZ.gpu_data, aux.gpu_data, size);
    cudaDeviceSynchronize();
}


#endif  // ZKRELU_CUH
