#include "src/runner.cuh"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<"\n"; return 2;} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS error: "<<e<<"\n"; return 2;} } while(0)

struct Args { int kernel=0,M=4096,N=4096,K=4096,seed=1234,device=0,warmup=10,iters=30; std::string mode="benchmark"; };
static int geti(char **&p, char **end) { if (++p>=end) throw std::runtime_error("missing value"); return std::stoi(*p); }
static std::string gets(char **&p, char **end) { if (++p>=end) throw std::runtime_error("missing value"); return *p; }
static Args parse(int argc,char**argv){ Args a; for(char**p=argv+1;p<argv+argc;++p){std::string s=*p; if(s=="--kernel")a.kernel=geti(p,argv+argc); else if(s=="--M")a.M=geti(p,argv+argc); else if(s=="--N")a.N=geti(p,argv+argc); else if(s=="--K")a.K=geti(p,argv+argc); else if(s=="--seed")a.seed=geti(p,argv+argc); else if(s=="--device")a.device=geti(p,argv+argc); else if(s=="--warmup")a.warmup=geti(p,argv+argc); else if(s=="--iters")a.iters=geti(p,argv+argc); else if(s=="--mode")a.mode=gets(p,argv+argc); else throw std::runtime_error("unknown argument "+s);} return a; }

static bool exact_safe(const Args&a){
  if(a.kernel<=2) return true;
  if(a.kernel>=3 && a.kernel<=8) return a.M%64==0 && a.N%64==0 && a.K%8==0;
  if(a.kernel==9) return a.M%128==0 && a.N%128==0 && a.K%16==0;
  if(a.kernel==10) return a.M%128==0 && a.N%128==0 && a.K%16==0;
  if(a.kernel==11) return a.M%128==0 && a.N%256==0 && a.K%16==0;
  if(a.kernel==12) return a.M%128==0 && a.N%128==0 && a.K%16==0;
  return false;
}

static void cublas_ref(cublasHandle_t h,int M,int N,int K,float alpha,const float*A,const float*B,float beta,float*C){
  cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,B,CUDA_R_32F,N,A,CUDA_R_32F,K,&beta,C,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
static void naive_safe(int M,int N,int K,float alpha,const float*A,const float*B,float beta,float*C);

__global__ static void naive_safe_kernel(int M,int N,int K,float alpha,const float*A,const float*B,float beta,float*C){
  int r=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x; if(r>=M||c>=N)return; float s=0; for(int k=0;k<K;++k)s+=A[r*K+k]*B[k*N+c]; C[r*N+c]=alpha*s+beta*C[r*N+c];
}
static void naive_safe(int M,int N,int K,float alpha,const float*A,const float*B,float beta,float*C){dim3 b(16,16),g((N+15)/16,(M+15)/16); naive_safe_kernel<<<g,b>>>(M,N,K,alpha,A,B,beta,C);}

int main(int argc,char**argv){
  try {
    Args a=parse(argc,argv); if(a.kernel<0||a.kernel>12||a.M<=0||a.N<=0||a.K<=0) throw std::runtime_error("invalid dimensions/kernel");
    CUDA_CHECK(cudaSetDevice(a.device));
    size_t as=(size_t)a.M*a.K, bs=(size_t)a.K*a.N, cs=(size_t)a.M*a.N; std::mt19937 gen(a.seed); std::uniform_real_distribution<float>d(-1.f,1.f);
    std::vector<float> hA(as),hB(bs),hC(cs),hRef(cs); for(auto&x:hA)x=d(gen); for(auto&x:hB)x=d(gen); for(auto&x:hC)x=d(gen); hRef=hC;
    float *A,*B,*C,*R; CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&A),as*4)); CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&B),bs*4)); CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&C),cs*4)); CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&R),cs*4));
    CUDA_CHECK(cudaMemcpy(A,hA.data(),as*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(B,hB.data(),bs*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(C,hC.data(),cs*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(R,hC.data(),cs*4,cudaMemcpyHostToDevice));
    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h)); float alpha=1.f,beta=0.f; cublas_ref(h,a.M,a.N,a.K,alpha,A,B,beta,R); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(hRef.data(),R,cs*4,cudaMemcpyDeviceToHost));
    bool fallback=(a.kernel>0&&!exact_safe(a)); auto launch=[&](){ if(a.kernel==0)cublas_ref(h,a.M,a.N,a.K,alpha,A,B,beta,C); else if(fallback)naive_safe(a.M,a.N,a.K,alpha,A,B,beta,C); else run_kernel(a.kernel,a.M,a.N,a.K,alpha,A,B,beta,C,h); };
    CUDA_CHECK(cudaMemcpy(C,hC.data(),cs*4,cudaMemcpyHostToDevice)); for(int i=0;i<a.warmup;++i)launch(); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms; ms.reserve(a.iters); std::vector<cudaEvent_t> st(a.iters),en(a.iters); for(int i=0;i<a.iters;++i){CUDA_CHECK(cudaEventCreate(&st[i]));CUDA_CHECK(cudaEventCreate(&en[i]));}
    for(int i=0;i<a.iters;++i){CUDA_CHECK(cudaEventRecord(st[i]));launch();CUDA_CHECK(cudaEventRecord(en[i]));} CUDA_CHECK(cudaDeviceSynchronize()); for(int i=0;i<a.iters;++i){float t;CUDA_CHECK(cudaEventElapsedTime(&t,st[i],en[i]));ms.push_back(t);cudaEventDestroy(st[i]);cudaEventDestroy(en[i]);}
    CUDA_CHECK(cudaMemcpy(hC.data(),C,cs*4,cudaMemcpyDeviceToHost)); double maxAbs=0,maxRel=0,n2=0,ref2=0; long naninf=0; for(size_t i=0;i<cs;++i){double x=hC[i],y=hRef[i]; if(!std::isfinite(x)||!std::isfinite(y))naninf++; double ad=std::abs(x-y); maxAbs=std::max(maxAbs,ad); maxRel=std::max(maxRel,ad/std::max(std::abs(y),1e-3)); n2+=(x-y)*(x-y); ref2+=y*y;} double l2=std::sqrt(n2/(ref2+1e-30)); bool pass=maxAbs<1e-2&&l2<1e-4&&naninf==0;
    std::sort(ms.begin(),ms.end()); double mean=0;for(float x:ms)mean+=x;mean/=ms.size(); double var=0;for(float x:ms)var+=(x-mean)*(x-mean);var/=ms.size(); size_t p95i=(size_t)std::ceil(.95*ms.size())-1; double med=ms[ms.size()/2],p95=ms[p95i],tflops=(2.0*a.M*a.N*a.K)/(med*1e-3)/1e12;
    std::cout<<"kernel="<<a.kernel<<",M="<<a.M<<",N="<<a.N<<",K="<<a.K<<",seed="<<a.seed<<",device="<<a.device<<",fallback_naive="<<(fallback?1:0)<<",pass="<<(pass?1:0)<<",max_abs="<<maxAbs<<",max_rel="<<maxRel<<",norm_l2="<<l2<<",nan_inf="<<naninf<<",median_ms="<<med<<",mean_ms="<<mean<<",p95_ms="<<p95<<",std_ms="<<std::sqrt(var)<<",tflops="<<tflops<<"\n";
    cublasDestroy(h); cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(R); return pass?0:1;
  } catch(const std::exception&e){std::cerr<<"argument/runtime error: "<<e.what()<<"\n"; return 2;}
}
