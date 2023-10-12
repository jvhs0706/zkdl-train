#ifndef ZKRELU_CUH
#define ZKRELU_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 

const uint zkReLU_Q = 32;
const uint zkReLU_R = 16;
const uint zkReLU_B = 48;
const uint zkReLU_UB = 64;

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
    #pragma unroll
    for (uint i = zkReLU_B; i < zkReLU_UB; ++ i) {
        aux_ptr[gid * zkReLU_UB + i] = blstrs__scalar__Scalar_ZERO;
        aux_ptr[(gid + n) * zkReLU_UB + i] = blstrs__scalar__Scalar_ZERO;
    }
}

class zkReLU {
public:
    const uint size;
    FrTensor Z;
    FrTensor GA;

    FrTensor A;
    FrTensor GZ;
    
    FrTensor aux;
    G1TensorJacobian* com_ptr;

    zkReLU(const FrTensor& Z, const FrTensor& GA, const Commitment& com);
    ~zkReLU() {delete com_ptr;}
    void prove(const Commitment& gen);
};


zkReLU::zkReLU(const FrTensor& Z, const FrTensor& GA, const Commitment& gen): size(Z.size), Z(Z), GA(GA), GZ(Z.size), A(GA.size), aux(2 * Z.size * zkReLU_UB), com_ptr(nullptr)
{
    // make sure the four inputs are of the same size
    if (GA.size != size) throw std::invalid_argument("Z and GA must be of the same size");
    zkReLU_init_kernel<<<(size+FrNumThread-1)/FrNumThread,FrNumThread>>>(Z.gpu_data, GA.gpu_data, A.gpu_data, GZ.gpu_data, aux.gpu_data, size);
    cudaDeviceSynchronize();
    this -> Z.mont();
    this -> GA.mont();
    this -> GZ.mont();
    this -> A.mont();
    com_ptr = new G1TensorJacobian(gen.commit(aux));
    cout << "Commitment size:" << com_ptr -> size << endl;
}

KERNEL void s_kernel(Fr_t* s, Fr_t r, Fr_t r_)
{   
    const uint gid = GET_GLOBAL_ID();
    if (gid >= zkReLU_UB) return;

    s[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(long_to_scalar(1L << gid)), r);
    
    if (gid == zkReLU_R - 1)
    {
        s[gid] = blstrs__scalar__Scalar_add(s[gid], r_);
    }
    else if (gid <= zkReLU_B - 2)
    {
        auto temp = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(long_to_scalar(1L << (gid - zkReLU_R))), r_);
        s[gid] = blstrs__scalar__Scalar_add(s[gid], temp);
    }
    else if (gid == zkReLU_B - 1)
    {
        auto temp = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_mont(long_to_scalar(-1L << (gid - zkReLU_R))), r_);
        s[gid] = blstrs__scalar__Scalar_sub(temp, s[gid]);
    }
    else 
    {
        s[gid] = {0, 0, 0, 0, 0, 0, 0, 0};
    }
}

void zkReLU::prove(const Commitment& gen)
{
    auto& com = *com_ptr;
    vector <Fr_t> proof;
    // cout << aux.size << " " << u_bin.size() << " " << v_bin.size() << endl;
    
    const vector<Fr_t>& u_bin = random_vec(ceilLog2(size) + 7);
    // const vector<Fr_t> u(u_bin.begin() + 6, u_bin.end() - 1);
    auto bin_proof = binary_sumcheck(aux, random_vec(ceilLog2(size) + 7), u_bin);

    auto temp_r = random_vec(2);
    auto r = temp_r[0], r_ = temp_r[1];
    FrTensor s(zkReLU_UB);
    s_kernel<<<1,zkReLU_UB>>>(s.gpu_data, r, r_);
    cudaDeviceSynchronize();

    auto aux_ = aux.partial_me({u_bin.begin() + 6, u_bin.end()}, zkReLU_UB);
    auto proof_recover = inner_product_sumcheck(aux_, s, {u_bin.begin(), u_bin.begin() + 6});
    proof.insert(proof.end(), bin_proof.begin(), bin_proof.end());
    proof.insert(proof.end(), proof_recover.begin(), proof_recover.end());
    
    cout << "zkReLU sumcheck proof size = " << proof.size() << endl;
    gen.open(aux, com, u_bin);

    // s_kernel<<<1,zkReLU_UB>>>(s.gpu_data, {0, 0, 0, 0, 0, 0, 0, 0}, {4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881});
    // cudaDeviceSynchronize();
    // cout << s.unmont() << endl;
    // s_kernel<<<1,zkReLU_UB>>>(s.gpu_data, {4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881}, {0, 0, 0, 0, 0, 0, 0, 0});
    // cudaDeviceSynchronize();
    // cout << s.unmont() << endl;
}

// KERNEL void zkReLU_phase1_kernel(GLOBAL Fr_t* Z_ptr, GLOBAL Fr_t* GA_ptr, GLOBAL Fr_t* A_ptr, GLOBAL Fr_t* GZ_ptr, GLOBAL Fr_t* aux_ptr, 
//     GLOBAL Fr_t* out0, GLOBAL Fr_t* out1, GLOBAL Fr_t* out2, Fr_t r, Fr_t r_, Fr_t w, uint in_size, uint out_size)
// {
//     const uint gid = GET_GLOBAL_ID();
//     if (gid >= out_size) return;

//     uint gid0 = 2 * gid;
//     uint gid1 = 2 * gid + 1;

//     Fr_t z0 = Z_ptr[gid0];
//     Fr_t z1 = blstrs__scalar__Scalar_sub(Z_ptr[gid1], Z_ptr[gid0]) if (gid1 < in_size) else blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ZERO, Z_ptr[gid0]);

//     Fr_t a0 = blstrs__scalar__Scalar_mul
//     Fr_t a1 = blstrs__scalar__Scalar_sub(Z_ptr[gid1], Z_ptr[gid0]);



//     out0[gid] = blstrs__scalar__Scalar_mul(Z_ptr[gid0], r);
//     out1[gid] = blstrs__scalar__Scalar_mul(Z_ptr[gid1], r);
//     out2[]
// }


#endif  // ZKRELU_CUH
