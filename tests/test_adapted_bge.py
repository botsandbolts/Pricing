"""Release selection and GUI transform invariants; no downloaded models required."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from pricing.adapted_bge import AdaptedProjector
from scripts.publish_adapted_bge import choose

class GridSelection(unittest.TestCase):
    def fixture(self,root):
        for i,(d,n,m) in enumerate((d,n,m) for d in (32,64) for n in (30,100) for m in (.1,.3)):
            folder=Path(root)/f'adapted-config-{d}-{n}-{m}'/'ask'
            (folder/'bayesian').mkdir(parents=True)
            (folder/'features.json').write_text(json.dumps(dict(umap_dimensions=d,n_neighbors=n,min_dist=m,n_epochs=500,metric='cosine',encoder_bundle={'identity':'checkpoint0'})))
            report=dict(selected_likelihood='student_t',candidates=[dict(likelihood='student_t',validation_mae_usd=100+i,max_split_rhat=1.01)],point_scores={'mae_usd':100-i},test_coverage=.9)
            (folder/'bayesian/bayesian-evaluation.json').write_text(json.dumps(report))
    def test_validation_selects_even_when_test_ranks_opposite(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp);best,entries=choose(tmp)
            self.assertEqual(best['validation_mae_usd'],100)
            self.assertEqual(best['test_mae_usd'],100)
            self.assertEqual(len(entries),8)
    def test_missing_configuration_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            import shutil
            shutil.rmtree(next(Path(tmp).iterdir()))
            with self.assertRaises(ValueError):choose(tmp)
    def test_mixed_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp);path=next(Path(tmp).glob('*/ask/features.json'))
            spec=json.loads(path.read_text());spec['encoder_bundle']['identity']='other';path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):choose(tmp)

class GUIFeatures(unittest.TestCase):
    def test_unique_descriptions_transformed_and_zero_columns_retained(self):
        calls=[]
        class Encoder:
            def encode(self,texts):
                calls.append(('encode',texts))
                return np.ones((len(texts),384),dtype='float32')
        class Reducer:
            n_components=2
            def transform(self,vectors):
                calls.append(('transform',len(vectors)))
                return np.tile([0.,2.],(len(vectors),1))
        projector=AdaptedProjector.__new__(AdaptedProjector)
        projector.encoder=Encoder();projector.reducer=Reducer();projector.cache={}
        projector.mini=SimpleNamespace(transform=lambda texts:csr_matrix(np.full((len(texts),1),100.)))
        matrix=projector.transform(['red sedan $5000','red sedan $5000'])
        self.assertEqual(matrix.shape,(2,3))
        self.assertEqual(matrix.nnz,6) # explicit zero is a numerical feature, not sparse missing
        np.testing.assert_allclose(matrix.toarray(),[[100,0,2],[100,0,2]])
        np.testing.assert_allclose(projector.transform(['red sedan $5000']).toarray(),[[100,0,2]])
        self.assertEqual(len(calls),2)
        self.assertNotIn('$5000',calls[0][1][0])
        self.assertEqual(projector.get_feature_names_out().tolist(),['MiniLM description price','bge_umap_0','bge_umap_1'])

if __name__=='__main__':unittest.main()
