#ifndef ZK_MATMUL_H
#define ZK_MATMUL_H
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "g1-tensor.cuh"
#include "proof.cuh"
#include "timer.hpp"
#include "commitment.cuh"

class zkMatMul {
public:
    FrTensor A, B;
    uint num, m, n, k;
    G1TensorJacobian comA, comB;
    zkMatMul(const FrTensor& A, const FrTensor& B, uint num, uint m, uint n, uint k, Commitment& genA, Commitment& genB): A(A), B(B), num(num), m(m), n(n), k(k), comA(genA.commit(A)), comB(genB.commit(B)) {
        if (A.size != num * m * n || B.size != num * n * k) throw std::runtime_error("size mismatch");
        cout << "Commitment size: " << comA.size + comB.size << endl;
    }

    static std::pair<FrTensor, FrTensor> reduce(const FrTensor& A, const FrTensor& B, uint num, uint m, uint n, uint k);
    static std::pair<FrTensor, FrTensor> phase1(const FrTensor& A_reduced, const FrTensor& B_reduced, uint num, uint n, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof);
    void prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_m, const vector<Fr_t>& u_n, const vector<Fr_t>& u_k, const Commitment& genA, const Commitment& genB);
};

// KERNEL void zkMatMul_phase1_kernel(GLOBAL Fr_t* Z_ptr, GLOBAL Fr_t* GA_ptr, GLOBAL Fr_t* A_ptr, GLOBAL Fr_t* GZ_ptr, GLOBAL Fr_t* aux_ptr, uint n)
// {
//     const uint gid = GET_GLOBAL_ID();
//     if (gid >= n) return;
//     long z = scalar_to_long(Z_ptr[gid]);
//     long ga = scalar_to_long(GA_ptr[gid]);
//     bool mask = z >= 0;
//     A_ptr[gid] = long_to_scalar(rescale(mask * z));
//     GZ_ptr[gid] = long_to_scalar(rescale(mask * ga)); 
//     #pragma unroll
//     for (uint i = 0; i < zkReLU_B; ++ i) {
//         aux_ptr[gid * zkReLU_B + i] = ((z >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
//         aux_ptr[(gid + n) * zkReLU_B + i] = ((ga >> i) & 1) ? blstrs__scalar__Scalar_ONE: blstrs__scalar__Scalar_ZERO;
//     }
// }

KERNEL void zkMatMul_phase1_step(GLOBAL Fr_t *a, GLOBAL Fr_t *b, GLOBAL Fr_t *out0, GLOBAL Fr_t *out1, GLOBAL Fr_t *out2, uint in_num, uint out_num, uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= out_num * n) return;
    
    uint gid_num = gid / n;
    uint gid_n = gid % n;
    uint gid0 = 2 * gid_num + gid_n;
    uint gid1 = 2 * (gid_num + 1) + gid_n;
    uint in_size = in_num * n;

    Fr_t a0 = (gid0 < in_size) ? a[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t b0 = (gid0 < in_size) ? b[gid0] : blstrs__scalar__Scalar_ZERO;
    Fr_t a1 = (gid1 < in_size) ? a[gid1] : blstrs__scalar__Scalar_ZERO;
    Fr_t b1 = (gid1 < in_size) ? b[gid1] : blstrs__scalar__Scalar_ZERO;
    out0[gid] = blstrs__scalar__Scalar_mul(a0, b0);
    out1[gid] = blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(a0, blstrs__scalar__Scalar_sub(b1, b0)), 
        blstrs__scalar__Scalar_mul(b0, blstrs__scalar__Scalar_sub(a1, a0)));
    out2[gid] = blstrs__scalar__Scalar_mul(blstrs__scalar__Scalar_sub(a1, a0), blstrs__scalar__Scalar_sub(b1, b0));
}

std::pair<FrTensor, FrTensor> zkMatMul::phase1(const FrTensor& A_reduced, const FrTensor& B_reduced, uint num, uint n, vector<Fr_t>::const_iterator u_begin, vector<Fr_t>::const_iterator u_end, vector<Fr_t>::const_iterator v_begin, vector<Fr_t>::const_iterator v_end, vector<Fr_t>& proof)
{
    if (v_end - v_begin != u_end - u_begin) throw std::runtime_error("Incompatible dimensions 6");
    if (v_begin >= v_end) {
        return {A_reduced, B_reduced};
    }

    auto out_num = (num + 1) / 2;
    auto in_size = num * n;
    auto out_size = out_num * n;
    FrTensor out0(out_num * n), out1(out_num * n), out2(out_num * n);
    zkMatMul_phase1_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(A_reduced.gpu_data, B_reduced.gpu_data, out0.gpu_data, out1.gpu_data, out2.gpu_data, num, out_num, n);
    cudaDeviceSynchronize();
    vector<Fr_t> u_(u_begin + 1, u_end);
    proof.push_back(out0.partial_me(u_, n).sum());
    proof.push_back(out1.partial_me(u_, n).sum());
    proof.push_back(out2.partial_me(u_, n).sum());

    FrTensor a_new(out_size), b_new(out_size);
    Fr_partial_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(A_reduced.gpu_data, a_new.gpu_data, *v_begin, in_size, out_size, n);
    cudaDeviceSynchronize();
    Fr_partial_me_step<<<(out_size+FrNumThread-1)/FrNumThread,FrNumThread>>>(B_reduced.gpu_data, b_new.gpu_data, *v_begin, in_size, out_size, n);
    cudaDeviceSynchronize();
    return phase1(a_new, b_new, out_num, n, u_begin + 1, u_end, v_begin + 1, v_end, proof);
}

void zkMatMul::prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_m, const vector<Fr_t>& u_n, const vector<Fr_t>& u_k, const Commitment& genA, const Commitment& genB)
{
    auto A_reduced = Fr_partial_me(A, u_m.begin(), u_m.end(), n); // num * n
    auto B_reduced = Fr_partial_me(B, u_k.begin(), u_k.end(), 1); // num * n
    vector<Fr_t> proof;
    auto phase1_out = zkMatMul::phase1(A_reduced, B_reduced, num, n, u_num.begin(), u_num.end(), v_num.begin(), v_num.end(), proof);
    auto& a = phase1_out.first;
    auto& b = phase1_out.second;
    auto phase_2_proof = inner_product_sumcheck(a, b, u_n);
    proof.insert(proof.end(), phase_2_proof.begin(), phase_2_proof.end());
    cout << "zkMatMul sumcheck proof size: " << proof.size() << endl;

    genA.open(A, comA, concatenate<Fr_t>({u_n, u_m, v_num}));
    genB.open(B, comB, concatenate<Fr_t>({u_k, u_n, v_num}));
}

#endif