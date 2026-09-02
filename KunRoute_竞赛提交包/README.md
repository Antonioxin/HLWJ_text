# 玄枢 KunRoute - 竞赛提交包

KunRoute 面向 5~10 个异构大模型，使用历史五元组：

`D = (query, model, quality_score, latency, cost)`

训练轻量级 Q/C/L 路由器。新请求按以下顺序选择目标模型：

1. `quality >= tau`
2. 在质量达标模型中选择最低成本
3. 成本进入等价区间时选择最低延迟

在线推理部署于鲲鹏 CPU；C++ 内核提供 AArch64 NEON 路径与 OpenMP 多核请求并行。

## 目录

- `docs/`：Word 与 PDF 技术方案
- `config/`：Model1~Model10 注册表与 Router 配置
- `data/`：标准五元组数据集，500 个 query × 10 个模型 = 5,000 条记录
- `src/`：数据生成、LightMLP 训练、在线路由、评估代码
- `cpp/`：鲲鹏 AArch64 NEON + OpenMP 推理内核
- `tests/`：单元测试
- `artifacts/`：训练权重、标准化参数、历史质量先验、质量校准表
- `results/`：标准算例、冻结测试评估、C++ 内核自检输出

## Python 一键运行

```bash
python src/generate_sample_data.py
python src/train_router.py
python src/evaluate.py
python src/router.py
python -m unittest discover -s tests -v
```

依赖：Python 3.10+、NumPy。

## 鲲鹏 CPU 构建

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
OMP_NUM_THREADS=1  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=2  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=4  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=8  ./build/kunroute_bench artifacts/weights.bin 100000
OMP_NUM_THREADS=16 ./build/kunroute_bench artifacts/weights.bin 100000
```

AArch64 构建使用 `-march=armv8-a+simd`。核心乘加使用 `arm_neon.h` 中的 NEON intrinsics；多核并行使用 OpenMP。
