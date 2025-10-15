from sklearn.base import BaseEstimator, ClassifierMixin
import dgl
import torch as th
from sklearn.metrics import accuracy_score


class LinkPredictor(BaseEstimator, ClassifierMixin):
    def __init__(self, feat_attr: str = 'feat'):
        self.feat_attr = feat_attr

    def fit(self, train_graph: dgl.DGLHeteroGraph):
        raise NotImplementedError

    def predict(self, train_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph, ) -> th.Tensor:
        raise NotImplementedError

    def score(self, train_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph,
              y: th.Tensor, scoring=accuracy_score):
        pred = self.predict(train_graph, indicator_graph)
        return scoring(y, pred)
