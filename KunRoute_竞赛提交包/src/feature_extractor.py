import math, re, hashlib
import numpy as np

TASKS = ["qa", "math", "code", "summary", "translation", "extraction"]
TASK_KEYWORDS = {
    "qa": ["为什么", "什么", "如何", "解释", "介绍", "question", "why", "what", "how"],
    "math": ["计算", "求解", "证明", "方程", "概率", "几何", "积分", "矩阵", "calculate", "solve", "equation"],
    "code": ["代码", "python", "java", "sql", "bug", "debug", "函数", "算法", "编程", "class", "function", "query"],
    "summary": ["总结", "摘要", "概括", "归纳", "提炼", "summary", "summarize"],
    "translation": ["翻译", "译成", "英文", "中文", "translate", "translation"],
    "extraction": ["提取", "抽取", "字段", "json", "表格", "结构化", "extract", "schema", "key-value"]
}

BASE_DIM = 16
TASK_DIM = 6
HASH_DIM = 32
QUERY_DIM = BASE_DIM + TASK_DIM + HASH_DIM
MODEL_DIM = 14  # cost, latency, 6 skill tags, 6 historical task priors
CROSS_DIM = 6
INPUT_DIM = QUERY_DIM + MODEL_DIM + CROSS_DIM

CODE_SET = set(b"{}[]();=<>_:#")
MATH_SET = set(b"+-*/%^=<>~")
PUNCT_SET = set(b".,!?;:'\"()[]{}<>-_/#@")


def _ratio(count, n):
    return float(count) / max(1, n)


def task_scores(text: str) -> np.ndarray:
    low = text.lower()
    scores = []
    for t in TASKS:
        s = 0.0
        for kw in TASK_KEYWORDS[t]:
            if kw in low:
                s += 1.0
        scores.append(s)
    arr = np.asarray(scores, dtype=np.float32)
    if arr.max() == 0:
        arr[0] = 1.0
    # soft-normalize; preserves multi-label signal
    arr = np.exp(np.minimum(arr, 4.0))
    arr /= arr.sum()
    return arr.astype(np.float32)


def dominant_task(text: str) -> str:
    return TASKS[int(np.argmax(task_scores(text)))]


def _hash_bigrams(data: bytes) -> np.ndarray:
    v = np.zeros(HASH_DIM, dtype=np.float32)
    if len(data) < 2:
        return v
    for i in range(len(data)-1):
        a, b = data[i], data[i+1]
        h = ((a * 1315423911) ^ (b * 2654435761) ^ (i * 97531)) & 0xffffffff
        idx = h % HASH_DIM
        sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
        v[idx] += sign
    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v /= norm
    return v


def extract_query_features(text: str) -> np.ndarray:
    data = text.encode("utf-8", errors="ignore")
    n = len(data)
    digit = sum(48 <= x <= 57 for x in data)
    alpha = sum((65 <= x <= 90) or (97 <= x <= 122) for x in data)
    space = data.count(32)
    newline = data.count(10)
    punct = sum(x in PUNCT_SET for x in data)
    code = sum(x in CODE_SET for x in data)
    mathc = sum(x in MATH_SET for x in data)
    quote = data.count(34) + data.count(39)
    comma = data.count(44)
    colon = data.count(58)
    json_hint = int((b"{" in data and b"}" in data) or (b"json" in data.lower()))
    url_hint = int(b"http://" in data.lower() or b"https://" in data.lower() or b"www." in data.lower())
    qmark = data.count(63) + text.count("？") + data.count(33) + text.count("！")
    line_count = text.count("\n") + 1
    token_approx = max(1, len(re.findall(r"\w+|[\u4e00-\u9fff]", text)))
    base = np.asarray([
        min(math.log1p(n) / 10.0, 1.0),
        _ratio(digit, n), _ratio(alpha, n), _ratio(space, n), _ratio(newline, n),
        _ratio(punct, n), _ratio(code, n), _ratio(mathc, n),
        float(json_hint), float(url_hint), min(qmark / 5.0, 1.0),
        _ratio(quote, n), _ratio(comma, n), _ratio(colon, n),
        min(line_count / 20.0, 1.0), min(token_approx / 512.0, 1.0)
    ], dtype=np.float32)
    return np.concatenate([base, task_scores(text), _hash_bigrams(data)]).astype(np.float32)


def model_vector(row: dict, task_priors: dict) -> np.ndarray:
    # registry values are normalized service inputs; latency is scaled to seconds-like magnitude
    cost = float(row["unit_cost"]) / 8.0
    latency = float(row["avg_latency_ms"]) / 2000.0
    skill = np.asarray([float(row[f"skill_{t}"]) for t in TASKS], dtype=np.float32)
    pri = np.asarray([float(task_priors[row["model_id"]].get(t, 0.8)) for t in TASKS], dtype=np.float32)
    return np.concatenate([[cost, latency], skill, pri]).astype(np.float32)


def pair_features(query: str, model_row: dict, task_priors: dict) -> np.ndarray:
    q = extract_query_features(query)
    mv = model_vector(model_row, task_priors)
    t = q[BASE_DIM:BASE_DIM+TASK_DIM]
    skill = mv[2:2+TASK_DIM]
    cross = (t * skill).astype(np.float32)
    x = np.concatenate([q, mv, cross]).astype(np.float32)
    assert x.shape[0] == INPUT_DIM
    return x
