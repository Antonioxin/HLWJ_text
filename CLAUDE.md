# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

所有实际代码位于 `KunRoute_竞赛提交包/` 子目录中；顶层还有该技术方案的 `.docx` / `.pdf` 副本。下列命令均需在 `KunRoute_竞赛提交包/` 目录下执行。脚本通过 `Path(__file__).resolve().parents[1]` 定位根目录，模块导入依赖运行目录（`python src/x.py` 会把 `src/` 加入 sys.path），因此不要改变目录结构。

## 项目概览

玄枢 KunRoute：面向 5~10 个异构大模型（Model1~Model10，质量/成本/延迟递增）的请求路由系统。训练一个轻量 MLP（称 Q/C/L 路由器），给定 query 与模型，预测三要素：quality、cost、latency。新请求据此选出满足质量阈值、成本尽量低、成本等价区间内延迟最低的模型。另有 C++ 内核用于鲲鹏 AArch64 部署（NEON + OpenMP）。

核心数据模型是历史五元组 `(query, model, quality_score, latency, cost)`，由 `generate_sample_data.py` 确定性生成：500 个 query × 10 模型 = 5000 行。

## 常用命令

```bash
python src/generate_sample_data.py   # 生成 data/sample_history.csv（确定性，SEED=20260901）
python src/train_router.py           # 训练并写入 artifacts/
python src/evaluate.py               # 冻结测试集评估，写入 results/sample_evaluation.json
python src/router.py                 # 单条路由演示，输出 scores 与 route 决策
python -m unittest discover -s tests -v   # 运行全部单测（tests/test_router.py）
./run_all.sh                         # 端到端：数据→训练→评估→路由→单测→C++ 构建与 bench
```

鲲鹏 / AArch64 C++ 构建与基准（`cpp/` 源码，产物 `build/kunroute_bench`，输入为 `artifacts/weights.bin`）：

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
OMP_NUM_THREADS=N ./build/kunroute_bench artifacts/weights.bin 100000
```

依赖仅 `numpy>=1.24`（见 `requirements.txt`），无需 GPU。

## 目录结构与数据流

`src/` 各脚本按 README 顺序构成流水线：

- `generate_sample_data.py` → `data/sample_history.csv`
- `train_router.py` 读取 `data/sample_history.csv` + `config/model_registry.csv` + `config/router_config.json` → 写出 `artifacts/`（`router_weights.npz`、`weights.bin`、`model_priors.json`、`calibration.json`、`model_meta.json`、`split.json`）
- `router.py`（运行时，仅推理，不依赖训练代码）读取 `config/` + `artifacts/`，导出 `KunRoute` 类
- `evaluate.py` 加载 `KunRoute` + `artifacts/split.json`，在测试 query 上对比"oracle"（用 ground-truth 的后验最优选择），写入 `results/`
- `tests/test_router.py` 通过插入 `src/` 到 sys.path 导入 `feature_extractor` 与 `router`

## 关键架构细节

- **特征工程** (`feature_extractor.py`)：`INPUT_DIM = 74` = 查询 54 维（16 字符统计基元 + 6 维 task softmax 关键词打分 + 32 维哈希 bigram 归一化嵌入）+ 模型 14 维（归一化 cost/latency + 6 个 skill 标签 + 6 个历史 task 先验）+ 6 维 task×skill 交叉。6 个任务类型固定为 `TASKS = ["qa","math","code","summary","translation","extraction"]`。CSV 读取统一用 `encoding='utf-8-sig'` 剥离 BOM。
- **训练** (`train_router.py`)：单隐层 ReLU MLP（hidden=48），Adam，输出 3 个量：quality（`clip` 到 [0,1]）、`log1p(cost)`、`log1p(latency/1000)`；特征与标签按训练集均值/标准差标准化。query 级 70/15/15 划分（`split_queries`），按 val loss 早停。训练后计算每 model×task 的校准余量 `calibration.json`：质量正向过预测残差（`qhat - q` 截断到 ≥0）的 `quality_margin_quantile`（默认 0.90）分位数。
- **推理与决策** (`router.py`)：`_predict_pair` 输出 4 个质量量——`pred_quality`（MLP 预测与在线先验按 `online_quality_blend` 加权）、`quality_safe`（再减去该校准余量）。`route()` 逻辑：`quality_safe >= tau`（默认 0.8）为可行集 → 成本 <= `cmin*(1+cost_tie_epsilon)` 为等价带 → 取其中延迟最低者；无可行解时回退到最高 `quality_safe` 并置 `constraint_unmet=True`。`update_online()` 用 EMA（默认 α=0.15）把新观测融合进在线质量/延迟估计，用于真实部署中的持续校准。
- **权重二进制格式** (`weights.bin`)：`train_router.write_weights` 写 `<4sIII>` 魔数 `KRT1` + din/dh/dout，随后按行优先依次写 W1,b1,W2,b2 的 float32。C++ 内核 `router_kernel.cpp` 用同一布局加载并复现 MLP 前向。
- **C++ 内核** (`cpp/router_kernel.cpp`)：`__aarch64__` 分支用 `arm_neon.h`（`vld1q_f32`/`vmlaq_f32`/`vaddvq_f32`）做核心乘加，其余平台回退标量；`#pragma omp parallel for` 做请求级并行；输出 path/threads/requests/seconds/qps/checksum。`CMakeLists.txt` 在 AArch64 上加 `-march=armv8-a+simd`。
- **配置集中处** (`config/router_config.json`)：路由阈值、校准分位数、成本等价 ε、MLP 超参、seed、在线混合比例。所有随机流程均以 `seed=20260901` 确定性复现。

## 风格约定

- 代码风格高度紧凑：单字母变量、无 docstring、长单行表达式，中文注释稀缺。修改时保持该风格，不必补注释或扩充。
- 用户面向的文档/README/技术方案与路径名均为中文；标识符、字段名、JSON key 为英文（如 `quality_score`、`skill_qa`）。
- 人为拆分保留 70/15/15 与评估指标含义：评估关注 `quality_pass_rate`、`oracle_route_accuracy`、`relative_cost_vs_all_strong`、`selected_latency_p95` 等（见 `results/sample_evaluation.json`）。改动前向/特征维度时，`INPUT_DIM`、`router_weights.npz` 与 `weights.bin`、C++ 内核需同步更新，且 `test_query_feature_shape`/`test_pair_feature_shape` 断言（54 / INPUT_DIM）会保护该一致性。

## 项目文本写作要求

撰写面向评审/用户的**项目文本**（技术方案、提交文档、README、答辩/申报材料、商业计划书等）时，须遵守下列要求：

- **参照格式范本**：写作前必须先通读仓库顶层 `格式参考文本_行文结构.md`（本项目《格式参考文本.pdf》的九章行文结构提取），并按其中的卷面与分页框架、标题层级与编号、`图x-y`/`表x-y` 连续编号、图表配套规律及各章写作套路来组织文本。
- **写作风格**：专业、详实；突出技术创新点（把"新在哪里、难在哪里、验证了什么、带来什么收益"讲清楚），避免泛泛而谈；引用的技术事实（特征维度、模型结构、损失权重、校准分位、超参、路由决策规则等）均以 `KunRoute_竞赛提交包/` 的真实代码与配置为准。
- **逻辑严谨闭环**：每个论断遵循"论点 → 机制/方法 → 证据（指标/算例/代码） → 小结"的闭环；前后口径一致、章节互证，不出现无依据的断言或前后矛盾的数据。
- **内容必须严格贴合代码**：禁止虚构/杜撰不存在于 `src/`、`cpp/`、`config/`、`data/`、`results/`、`tests/` 的功能、公式、超参、流程或数字。
  - 指标只能引用 `results/*.json`、`results/kernel_selfcheck.txt`、`results/standard_case.json` 等实际输出；
  - 流程只能对应 `run_all.sh` 与 `src/*.py` 的真实调用链；
  - 模型池/超参只能对应 `config/model_registry.csv` 与 `config/router_config.json`；
  - 架构/伪代码必须能还原到 `src/` 与 `cpp/router_kernel.cpp` 的实现。
  - 文本中确需出现的占位/待补项（如市场数字、定价、财务、成员资料等仓库内不存在的内容）须显式标注 `【待补】`，不得自行编造。
