import csv, json, math
from pathlib import Path
import numpy as np
from feature_extractor import TASKS, pair_features, dominant_task

ROOT = Path(__file__).resolve().parents[1]

class KunRoute:
    def __init__(self, root=ROOT):
        self.root=Path(root)
        with (self.root/'config/model_registry.csv').open(encoding='utf-8-sig') as f:
            self.models=list(csv.DictReader(f))
        self.model_map={m['model_id']:m for m in self.models}
        self.cfg=json.loads((self.root/'config/router_config.json').read_text(encoding='utf-8'))
        self.priors=json.loads((self.root/'artifacts/model_priors.json').read_text(encoding='utf-8'))
        self.calibration=json.loads((self.root/'artifacts/calibration.json').read_text(encoding='utf-8'))
        z=np.load(self.root/'artifacts/router_weights.npz')
        self.W1=z['W1']; self.b1=z['b1']; self.W2=z['W2']; self.b2=z['b2']
        self.xmean=z['xmean']; self.xstd=z['xstd']; self.ymean=z['ymean']; self.ystd=z['ystd']
        self.online_quality={m['model_id']:{t:self.priors[m['model_id']][t] for t in TASKS} for m in self.models}
        self.online_latency={m['model_id']:float(m['avg_latency_ms']) for m in self.models}

    def _predict_pair(self, query, m):
        x=pair_features(query,m,self.priors)
        xn=(x-self.xmean)/self.xstd
        h=np.maximum(xn@self.W1+self.b1,0)
        y=(h@self.W2+self.b2)*self.ystd+self.ymean
        q=float(np.clip(y[0],0,1))
        c=float(max(1e-6,np.expm1(y[1])))
        l=float(max(1.0,np.expm1(y[2])*1000.0))
        task=dominant_task(query)
        qb=float(self.cfg['online_quality_blend']); lb=float(self.cfg['online_latency_blend'])
        q_online=(1-qb)*q + qb*float(self.online_quality[m['model_id']][task])
        l_online=(1-lb)*l + lb*float(self.online_latency[m['model_id']])
        margin=float(self.calibration[m['model_id']][task])
        q_safe=max(0.0,min(1.0,q_online-margin))
        return {'model':m['model_id'],'task':task,'pred_quality':q_online,'quality_safe':q_safe,'pred_cost':c,'pred_latency':l_online}

    def score_all(self, query):
        return [self._predict_pair(query,m) for m in self.models]

    def route(self, query, quality_threshold=None):
        tau=float(self.cfg['quality_threshold'] if quality_threshold is None else quality_threshold)
        scores=self.score_all(query)
        feasible=[s for s in scores if s['quality_safe']>=tau]
        if not feasible:
            best=max(scores,key=lambda s:(s['quality_safe'],-s['pred_cost'],-s['pred_latency']))
            return {**best,'constraint_unmet':True,'reason':'HIGHEST_QUALITY'}
        cmin=min(s['pred_cost'] for s in feasible)
        eps=float(self.cfg['cost_tie_epsilon'])
        cost_group=[s for s in feasible if s['pred_cost']<=cmin*(1.0+eps)]
        best=min(cost_group,key=lambda s:(s['pred_latency'],-s['quality_safe']))
        return {**best,'constraint_unmet':False,'reason':'QUALITY_PASS_MIN_COST'}

    def update_online(self, model_id, query, quality_score=None, latency=None, alpha=0.15):
        task=dominant_task(query)
        if quality_score is not None:
            old=self.online_quality[model_id][task]
            self.online_quality[model_id][task]=(1-alpha)*old+alpha*float(quality_score)
        if latency is not None:
            old=self.online_latency[model_id]
            self.online_latency[model_id]=(1-alpha)*old+alpha*float(latency)

if __name__=='__main__':
    r=KunRoute()
    q='请用 Python 编写函数，对整数列表去重并保持原顺序，同时说明时间复杂度。'
    print(json.dumps({'query':q,'route':r.route(q,0.8),'scores':r.score_all(q)},ensure_ascii=False,indent=2))
