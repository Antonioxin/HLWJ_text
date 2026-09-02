import csv, json, math, struct
from pathlib import Path
import numpy as np
from feature_extractor import TASKS, INPUT_DIM, pair_features, dominant_task

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sample_history.csv'
REG = ROOT / 'config' / 'model_registry.csv'
CFG = ROOT / 'config' / 'router_config.json'
ART = ROOT / 'artifacts'


def load_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def build_priors(rows, models):
    sums={(m['model_id'],t):[0.0,0] for m in models for t in TASKS}
    for r in rows:
        t=dominant_task(r['query'])
        k=(r['model'],t); sums[k][0]+=float(r['quality_score']); sums[k][1]+=1
    pri={}
    for m in models:
        mid=m['model_id']; pri[mid]={}
        for t in TASKS:
            s,n=sums[(mid,t)]
            pri[mid][t]=(s/n if n else 0.8)
    return pri


def split_queries(rows, seed):
    qs=sorted(set(r['query'] for r in rows))
    rng=np.random.default_rng(seed); rng.shuffle(qs)
    n=len(qs); ntr=int(n*0.70); nva=int(n*0.15)
    tr=set(qs[:ntr]); va=set(qs[ntr:ntr+nva]); te=set(qs[ntr+nva:])
    return tr,va,te


def dataset(rows, model_map, priors, allowed):
    X=[]; Y=[]; mids=[]; tasks=[]
    for r in rows:
        if r['query'] not in allowed: continue
        m=model_map[r['model']]
        X.append(pair_features(r['query'],m,priors))
        q=float(r['quality_score']); c=float(r['cost']); l=float(r['latency'])
        Y.append([q, math.log1p(c), math.log1p(l/1000.0)])
        mids.append(r['model']); tasks.append(dominant_task(r['query']))
    return np.asarray(X,np.float32), np.asarray(Y,np.float32), mids,tasks


class TinyMLP:
    def __init__(self, din, dh, dout, seed):
        rng=np.random.default_rng(seed)
        self.W1=(rng.standard_normal((din,dh))*math.sqrt(2/din)).astype(np.float32)
        self.b1=np.zeros(dh,np.float32)
        self.W2=(rng.standard_normal((dh,dout))*math.sqrt(2/dh)).astype(np.float32)
        self.b2=np.zeros(dout,np.float32)
        self.params=[self.W1,self.b1,self.W2,self.b2]
        self.m=[np.zeros_like(p) for p in self.params]
        self.v=[np.zeros_like(p) for p in self.params]
        self.step=0
    def forward(self,X):
        z=X@self.W1+self.b1
        h=np.maximum(z,0)
        y=h@self.W2+self.b2
        return y,(X,z,h)
    def train_step(self,X,Y,lr,weights):
        P,cache=self.forward(X); xb,z,h=cache; B=X.shape[0]
        diff=(P-Y)*weights[None,:]
        loss=float(np.mean((P-Y)**2 * weights[None,:]))
        dP=2*diff/(B*Y.shape[1])
        dW2=h.T@dP; db2=dP.sum(0)
        dh=dP@self.W2.T; dz=dh*(z>0)
        dW1=xb.T@dz; db1=dz.sum(0)
        grads=[dW1,db1,dW2,db2]
        self.step+=1; b1=0.9; b2=0.999; eps=1e-8
        for i,(p,g) in enumerate(zip(self.params,grads)):
            self.m[i]=b1*self.m[i]+(1-b1)*g
            self.v[i]=b2*self.v[i]+(1-b2)*(g*g)
            mh=self.m[i]/(1-b1**self.step); vh=self.v[i]/(1-b2**self.step)
            p-=lr*mh/(np.sqrt(vh)+eps)
        return loss


def write_weights(path, W1,b1,W2,b2):
    with path.open('wb') as f:
        f.write(struct.pack('<4sIII',b'KRT1',W1.shape[0],W1.shape[1],W2.shape[1]))
        for arr in [W1,b1,W2,b2]:
            f.write(np.asarray(arr,np.float32).tobytes(order='C'))


def main():
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    rows=load_csv(DATA); models=load_csv(REG); mmap={m['model_id']:m for m in models}
    priors=build_priors(rows,models)
    trq,vaq,teq=split_queries(rows,int(cfg['seed']))
    Xtr,Ytr,_,_=dataset(rows,mmap,priors,trq)
    Xva,Yva,mva,tva=dataset(rows,mmap,priors,vaq)
    Xte,Yte,mte,tte=dataset(rows,mmap,priors,teq)
    xmean=Xtr.mean(0); xstd=Xtr.std(0)+1e-6
    ymean=Ytr.mean(0); ystd=Ytr.std(0)+1e-6
    Xtrn=(Xtr-xmean)/xstd; Xvan=(Xva-xmean)/xstd; Xten=(Xte-xmean)/xstd
    Ytrn=(Ytr-ymean)/ystd
    net=TinyMLP(INPUT_DIM,int(cfg['hidden_dim']),3,int(cfg['seed']))
    rng=np.random.default_rng(int(cfg['seed'])+1)
    bs=int(cfg['batch_size']); epochs=int(cfg['epochs']); lr=float(cfg['learning_rate'])
    out_weights=np.array([4.0,1.0,0.7],np.float32)
    best=None; best_loss=1e9; patience=35; bad=0
    for ep in range(epochs):
        idx=rng.permutation(len(Xtrn)); losses=[]
        for s in range(0,len(idx),bs):
            ii=idx[s:s+bs]
            losses.append(net.train_step(Xtrn[ii],Ytrn[ii],lr,out_weights))
        pv,_=net.forward(Xvan); pv=pv*ystd+ymean
        val=float(np.mean((pv-Yva)**2))
        if val < best_loss-1e-6:
            best_loss=val; best=[p.copy() for p in net.params]; bad=0
        else:
            bad+=1
            if bad>=patience: break
    if best:
        net.W1[:]=best[0]; net.b1[:]=best[1]; net.W2[:]=best[2]; net.b2[:]=best[3]
    # calibration: positive overprediction residual qhat - q by model/task
    pv,_=net.forward(Xvan); pv=pv*ystd+ymean
    residuals={m['model_id']:{t:[] for t in TASKS} for m in models}
    for i,(mid,t) in enumerate(zip(mva,tva)):
        residuals[mid][t].append(max(0.0,float(pv[i,0]-Yva[i,0])))
    qtile=float(cfg['quality_margin_quantile'])
    calibration={}
    for m in models:
        mid=m['model_id']; calibration[mid]={}
        allr=sum([residuals[mid][t] for t in TASKS],[])
        global_margin=float(np.quantile(allr,qtile)) if allr else 0.03
        for t in TASKS:
            vals=residuals[mid][t]
            calibration[mid][t]=float(np.quantile(vals,qtile)) if len(vals)>=5 else global_margin
    # test metrics for engineering self-check
    pt,_=net.forward(Xten); pt=pt*ystd+ymean
    q_mae=float(np.mean(np.abs(np.clip(pt[:,0],0,1)-Yte[:,0])))
    c_mae=float(np.mean(np.abs(np.expm1(pt[:,1])-np.expm1(Yte[:,1]))))
    l_mae=float(np.mean(np.abs(np.expm1(pt[:,2])*1000-np.expm1(Yte[:,2])*1000)))
    ART.mkdir(exist_ok=True)
    np.savez(ART/'router_weights.npz',W1=net.W1,b1=net.b1,W2=net.W2,b2=net.b2,xmean=xmean,xstd=xstd,ymean=ymean,ystd=ystd)
    write_weights(ART/'weights.bin',net.W1,net.b1,net.W2,net.b2)
    (ART/'model_priors.json').write_text(json.dumps(priors,ensure_ascii=False,indent=2),encoding='utf-8')
    (ART/'calibration.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')
    meta={
        'input_dim':INPUT_DIM,'hidden_dim':int(cfg['hidden_dim']),'output_dim':3,
        'xmean':xmean.tolist(),'xstd':xstd.tolist(),'ymean':ymean.tolist(),'ystd':ystd.tolist(),
        'quality_threshold':float(cfg['quality_threshold']),
        'test_query_count':len(teq),'train_query_count':len(trq),'valid_query_count':len(vaq),
        'self_check':{'quality_mae':q_mae,'cost_mae':c_mae,'latency_mae_ms':l_mae}
    }
    (ART/'model_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (ART/'split.json').write_text(json.dumps({'train':sorted(trq),'valid':sorted(vaq),'test':sorted(teq)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta['self_check'],ensure_ascii=False))

if __name__=='__main__':
    main()
