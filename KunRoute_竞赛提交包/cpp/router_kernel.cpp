#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif

struct Weights {
    uint32_t din=0, dh=0, dout=0;
    std::vector<float> W1,b1,W2,b2;
};

static bool load_weights(const std::string& path, Weights& w) {
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;
    char magic[4]; f.read(magic,4);
    if(std::strncmp(magic,"KRT1",4)!=0) return false;
    f.read(reinterpret_cast<char*>(&w.din),4);
    f.read(reinterpret_cast<char*>(&w.dh),4);
    f.read(reinterpret_cast<char*>(&w.dout),4);
    w.W1.resize((size_t)w.din*w.dh); w.b1.resize(w.dh);
    w.W2.resize((size_t)w.dh*w.dout); w.b2.resize(w.dout);
    f.read(reinterpret_cast<char*>(w.W1.data()), w.W1.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(w.b1.data()), w.b1.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(w.W2.data()), w.W2.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(w.b2.data()), w.b2.size()*sizeof(float));
    return (bool)f;
}

static inline float dot_scalar(const float* a,const float* b,size_t n){
    float s=0.f; for(size_t i=0;i<n;++i) s+=a[i]*b[i]; return s;
}

static inline float dot_fast(const float* a,const float* b,size_t n){
#ifdef __aarch64__
    float32x4_t acc=vdupq_n_f32(0.f); size_t i=0;
    for(;i+4<=n;i+=4){
        float32x4_t va=vld1q_f32(a+i), vb=vld1q_f32(b+i);
        acc=vmlaq_f32(acc,va,vb);
    }
    float s=vaddvq_f32(acc); for(;i<n;++i) s+=a[i]*b[i]; return s;
#else
    return dot_scalar(a,b,n);
#endif
}

static void infer_one(const Weights& w,const float* x,float* y){
    std::vector<float> h(w.dh);
    // W1 is row-major [din, dh]. Gather each hidden column. For this tiny router,
    // transpose-on-load is unnecessary; the compiler still vectorizes inner loops on AArch64.
    for(uint32_t j=0;j<w.dh;++j){
        float s=w.b1[j];
#ifdef __aarch64__
        float32x4_t acc=vdupq_n_f32(0.f); uint32_t i=0;
        for(;i+4<=w.din;i+=4){
            float32x4_t vx=vld1q_f32(x+i);
            float vals[4]={w.W1[(i+0)*w.dh+j],w.W1[(i+1)*w.dh+j],w.W1[(i+2)*w.dh+j],w.W1[(i+3)*w.dh+j]};
            float32x4_t vw=vld1q_f32(vals); acc=vmlaq_f32(acc,vx,vw);
        }
        s+=vaddvq_f32(acc); for(;i<w.din;++i) s+=x[i]*w.W1[(size_t)i*w.dh+j];
#else
        for(uint32_t i=0;i<w.din;++i) s+=x[i]*w.W1[(size_t)i*w.dh+j];
#endif
        h[j]=std::max(0.f,s);
    }
    for(uint32_t k=0;k<w.dout;++k){
        float s=w.b2[k];
        for(uint32_t j=0;j<w.dh;++j) s+=h[j]*w.W2[(size_t)j*w.dout+k];
        y[k]=s;
    }
}

int main(int argc,char** argv){
    if(argc<2){ std::cerr<<"usage: kunroute_bench <weights.bin> [requests]\n"; return 2; }
    Weights w; if(!load_weights(argv[1],w)){ std::cerr<<"cannot load weights\n"; return 3; }
    size_t N=(argc>=3)?std::stoull(argv[2]):100000;
    std::vector<float> X(N*(size_t)w.din),Y(N*(size_t)w.dout);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(float& v:X) v=dist(rng);
    auto t0=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)N;++i) infer_one(w,X.data()+i*w.din,Y.data()+i*w.dout);
    auto t1=std::chrono::steady_clock::now();
    double sec=std::chrono::duration<double>(t1-t0).count();
    double qps=N/sec;
#ifdef __aarch64__
    const char* path="AArch64 NEON";
#else
    const char* path="portable scalar";
#endif
    int threads=1;
#ifdef _OPENMP
    threads=omp_get_max_threads();
#endif
    std::cout<<"path="<<path<<" threads="<<threads<<" requests="<<N
             <<" seconds="<<sec<<" qps="<<qps<<" checksum="<<Y[0]<<"\n";
    return 0;
}
