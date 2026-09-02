import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from feature_extractor import extract_query_features, INPUT_DIM, pair_features
from router import KunRoute

class TestKunRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=KunRoute(ROOT)
    def test_query_feature_shape(self):
        self.assertEqual(extract_query_features('请总结这段文本').shape[0],54)
    def test_pair_feature_shape(self):
        m=self.r.models[0]
        self.assertEqual(pair_features('写一个 Python 函数',m,self.r.priors).shape[0],INPUT_DIM)
    def test_score_all(self):
        s=self.r.score_all('请翻译成英文：成本最低优先')
        self.assertEqual(len(s),10)
        self.assertTrue(all(0<=x['quality_safe']<=1 for x in s))
    def test_route_returns_model(self):
        d=self.r.route('从文本中提取姓名和公司，输出 JSON。',0.8)
        self.assertIn(d['model'],[m['model_id'] for m in self.r.models])
    def test_online_update(self):
        mid=self.r.models[0]['model_id']; q='请总结这段材料'
        before=self.r.online_latency[mid]
        self.r.update_online(mid,q,quality_score=0.9,latency=before*1.2,alpha=0.5)
        self.assertGreater(self.r.online_latency[mid],before)

if __name__=='__main__': unittest.main()
