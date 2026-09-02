#!/usr/bin/env bash
set -euo pipefail
python src/generate_sample_data.py
python src/train_router.py
python src/evaluate.py
python src/router.py > results/sample_routing.json
python -m unittest discover -s tests -v
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/kunroute_bench artifacts/weights.bin 100000
