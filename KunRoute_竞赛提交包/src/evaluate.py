import csv, json
from collections import defaultdict
from pathlib import Path
from router import KunRoute

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/sample_history.csv'
SPLIT=ROOT/'artifacts/split.json'
OUT=ROOT/'results/sample_evaluation.json'


def load_rows():
    with DATA.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def main():
    rows=load_rows(); split=json.loads(SPLIT.read_text(encoding='utf-8')); test=set(split['test'])
    byq=defaultdict(list)
    for x in rows:
        if x['query'] in test: byq[x['query']].append(x)
    router=KunRoute(); tau=0.8
    n=0; qual_ok=0; oracle_match=0; sel_cost=0.; oracle_cost=0.; sel_lat=[]; allstrong_cost=0.
    # strongest = highest average quality on test matrix
    model_q=defaultdict(list)
    for q,rs in byq.items():
        for r in rs: model_q[r['model']].append(float(r['quality_score']))
    strong=max(model_q,key=lambda m:sum(model_q[m])/len(model_q[m]))
    for q,rs in byq.items():
        truth={r['model']:r for r in rs}
        decision=router.route(q,tau)
        chosen=truth[decision['model']]
        feasible=[r for r in rs if float(r['quality_score'])>=tau]
        if feasible:
            cmin=min(float(r['cost']) for r in feasible)
            bests=[r for r in feasible if float(r['cost'])<=cmin*1.01]
            oracle=min(bests,key=lambda r:float(r['latency']))
        else:
            oracle=max(rs,key=lambda r:float(r['quality_score']))
        n+=1; qual_ok+=float(chosen['quality_score'])>=tau; oracle_match+=chosen['model']==oracle['model']
        sel_cost+=float(chosen['cost']); oracle_cost+=float(oracle['cost']); sel_lat.append(float(chosen['latency']))
        allstrong_cost+=float(truth[strong]['cost'])
    sel_lat.sort()
    result={
      'test_queries':n,
      'quality_pass_rate':qual_ok/n,
      'quality_violation_rate':1-qual_ok/n,
      'oracle_route_accuracy':oracle_match/n,
      'average_selected_cost':sel_cost/n,
      'average_oracle_cost':oracle_cost/n,
      'relative_cost_vs_all_strong':sel_cost/allstrong_cost,
      'selected_latency_p95':sel_lat[int(0.95*(len(sel_lat)-1))]
    }
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
