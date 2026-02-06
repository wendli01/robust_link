# The Missing Link: Joint Legal Citation Prediction using Heterogeneous Graph Enrichment

This repository accompanies the 2026 paper [Robust Generalizable Legal Citation Prediction](https://arxiv.org/abs/2602.04812).

It contains all code as well as experimental setups described in the paper including results with all visualizations as standalone `jupyter` notebooks.


If you use code, data or any results in this repository, please cite:

```bibtex
@misc{wendlinger2026robustgeneralizableheterogeneouslegal,
      title={Robust Generalizable Heterogeneous Legal Link Prediction}, 
      author={Lorenz Wendlinger and Simon Alexander Nonn and Abdullah Al Zubaer and Michael Granitzer},
      year={2026},
      eprint={2602.04812},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.04812}, 
}

```

## Experiments

Complete experiments are stored in the notebooks for [Legal link prediction](experiments/legal_link_prediction.ipynb), datasets are analyzed in [dataset statistics](experiments/dataset_stats.ipynb).

## Dataset

We use two legal citation graph datasets:

- OLD201k, published by Milz et al. in [Analysis of a German Legal Citation Network](https://www.scitepress.org/publishedPapers/2021/106508/pdf/index.html).
- LiO338k, published by Milz et al. in [Law in Order: An Open Legal Citation
Network for New Zealand](https://link.springer.com/content/pdf/10.1007/978-981-99-8696-5_15.pdf).

### Data Acquisition


#### OLD201k

```python

old_ref_fn = 'data/old/old_ref.pkl'

if not os.path.exists(old_ref_fn):
    with open(old_ref_fn, 'wb') as fb:
        G_ = data.from_neo4j()
        pickle.dump(G_, fb)
    
G201k, case_df, law_df = data.load_old_graph(old_ref_fn)

```

## Installation


Installation via the provided conda envirionment is encouraged.

> `conda env create -f robust_link.yml`


To replicate the experiments, [`jupyter`](https://jupyter.org/install) needs to be installed as well, e.g. with


> `conda install -c conda-forge notebook`
> 
> or 
> 
> `pip install jupyterlab`


## Usage


All models are implemented as `sklearn` compatible Estimators. The main model, `graph_models.Model` is a superset of all used approaches and can be configured via hyperparameters.
For more information, see the [main study](experiments/old_link_prediction.ipynb).


```python
from src import graph_models, util, data
import dgl
import pandas as pd

    
G201k : dgl.DGLHeteroGraph
case_df: pd.DataFrame



rhge = graph_models.Model(device=device, cat_all=True, edge_dropout=.5, decoder_asymm_interleave=True)
scores = util.date_score_lp(rhge, G201k, dates={'Case': case_df.date}, random_state=42)
print(scores.agg(['mean', 'std').round(3))
```
### Node Features

Node features are generated as sentence embeddings via the convenience `text_processing.DocumentSentenceTransformer`:

```python

dst = text_processing.DocumentSentenceTransformer(device='cuda:5', model_name='jinaai/jina-embeddings-v3', trust_remote_code=True).fit()

G201k.ndata['jina_v3'] = {'Case': th.Tensor(np.nan_to_num(dst.transform(case_texts), 0)),
                               'Law': th.Tensor(np.nan_to_num(dst.transform(law_texts), 0))}

```
