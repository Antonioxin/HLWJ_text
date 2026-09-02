# KunRoute 鲲鹏 CPU 构建与运行

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
OMP_NUM_THREADS=1  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=2  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=4  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=8  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=16 ./build/kunroute_bench artifacts/weights.bin 100000
```

AArch64 构建自动启用 `-march=armv8-a+simd`。`router_kernel.cpp` 在 AArch64 路径中使用 `arm_neon.h` 的 `vld1q_f32`、`vmlaq_f32`、`vaddvq_f32` 完成核心乘加；OpenMP 用于请求级多核并行。
