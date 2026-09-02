import csv, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'config' / 'model_registry.csv'
OUT = ROOT / 'data' / 'sample_history.csv'
SEED = 20260901
TASKS = ['qa','math','code','summary','translation','extraction']

TEMPLATES = {
'qa': [
'请解释{topic}的核心原理，并给出两个应用场景。',
'为什么{topic}在实际系统中重要？请用通俗语言回答。',
'比较{topic}与{topic2}的主要区别，并给出适用条件。'],
'math': [
'求解方程 {a}x + {b} = {c}，写出计算过程。',
'已知一组数据为 {a},{b},{c},{d}，计算平均值并说明步骤。',
'一个项目成功概率为0.{a}，连续两次至少成功一次的概率是多少？'],
'code': [
'用 Python 编写函数，将列表中的重复元素去除并保持原顺序。',
'分析下面 SQL 的性能问题并给出索引建议：SELECT * FROM orders WHERE user_id={a} ORDER BY created_at DESC;',
'修复这段伪代码中的边界错误，并说明时间复杂度。'],
'summary': [
'请把下面材料压缩成三点摘要：企业上线多模型后，需要在质量、成本和延迟之间进行路由选择。',
'总结以下会议纪要，保留结论、负责人和下一步动作。',
'将这段技术说明概括为不超过120字的摘要。'],
'translation': [
'将下面中文翻译成英文：系统应先满足质量阈值，再选择成本最低的模型。',
'Translate into Chinese: The router predicts quality, cost and latency for every candidate model.',
'把下面句子翻译成简洁、正式的英文：模型价格变化后，路由策略应自动调整。'],
'extraction': [
'从文本中提取姓名、公司、职位三个字段，并输出 JSON：张三就职于华星科技，担任研发经理。',
'从订单描述中抽取订单号、金额、日期，并按 JSON 返回：订单A{a}，金额{b}元，日期2026-09-{d:02d}。',
'把以下内容整理为 key-value：客户=远航制造；问题=接口超时；优先级=高。']
}
TOPICS = ['大模型路由','缓存机制','数据库索引','向量检索','并行计算','编译优化','成本控制','请求调度']


def load_models():
    with REG.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))


def make_query(task, idx, rng):
    tpl = rng.choice(TEMPLATES[task])
    base = tpl.format(
        topic=rng.choice(TOPICS), topic2=rng.choice(TOPICS),
        a=rng.randint(2,9), b=rng.randint(10,99), c=rng.randint(20,120), d=rng.randint(1,28)
    )
    extra = f' 要求包含{2 + idx % 4}个要点，控制在{80 + (idx*17)%220}字以内。'
    return base + extra


def main():
    rng = random.Random(SEED)
    models = load_models()
    rows=[]
    n_queries = 500
    for qi in range(n_queries):
        task = TASKS[qi % len(TASKS)]
        q = make_query(task, qi, rng)
        # latent difficulty from text/task pattern; fixed across all models for this query
        difficulty = 0.20 + 0.60 * ((qi * 37) % 100) / 100.0
        if task in ('math','code'):
            difficulty += 0.08
        difficulty = min(difficulty, 0.92)
        in_units = max(1.0, len(q.encode('utf-8')) / 120.0)
        for mi, m in enumerate(models):
            skill = float(m[f'skill_{task}'])
            general = 0.04 * mi
            noise = rng.gauss(0, 0.025)
            qscore = sigmoid(-0.18 + 3.25*skill - 1.85*difficulty + general + noise)
            qscore = max(0.0, min(1.0, qscore))
            unit = float(m['unit_cost'])
            # standardized per-request cost units
            output_factor = 0.55 + 0.85*difficulty + (0.30 if task in ('summary','translation') else 0.0)
            cost = unit * (0.45*in_units + output_factor) * (1.0 + rng.gauss(0,0.025))
            base_lat = float(m['avg_latency_ms'])
            latency = base_lat * (0.72 + 0.70*difficulty) * (1.0 + rng.gauss(0,0.055))
            rows.append({
                'query':q,
                'model':m['model_id'],
                'quality_score':f'{qscore:.4f}',
                'latency':f'{max(120.0,latency):.2f}',
                'cost':f'{max(0.0001,cost):.4f}'
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['query','model','quality_score','latency','cost'])
        w.writeheader(); w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {OUT}')

if __name__=='__main__':
    main()
