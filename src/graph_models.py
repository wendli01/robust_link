import os
import time
import warnings
from typing import Optional, Sequence, Dict, Union

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F
from dgl import function as dgl_fn
from dgl.nn.pytorch import GraphConv, GATConv, HeteroGraphConv, RelGraphConv, SAGEConv, HGTConv
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from torch import nn
from torch.nn import Dropout

from src import util
from src.base import LinkPredictor
from src.layers import MixedSequential, LinkDecoder, EGATConv


class Model(LinkPredictor):
    def __init__(self, layer_sizes: Sequence[int] = (256, 256, 256), epochs: int = 200, lr: float = .001,
                 feat_attr: str = 'jina_emb_8192', verbose: bool = False, device: Optional[str] = None,
                 decoder_asymmetric: bool = True, decoder_method: str = 'ip', dropout: Optional[int] = .2,
                 self_loop: bool = True, weight_decay: float = 1e-5, reverse_edges: bool = True,
                 exposed_feat=('case_court_type', 'law_code', 'case_court_state'), k: int = 1,
                 num_heads: Optional[int] = None, norm_rep: bool = False, norm_feat: bool = False,
                 norm_all: bool = False, residual: bool = True, normalizer_feat: Optional[Sequence[str]] = None,
                 homogeneous: bool = False, random_state: int = 42, variational: bool = False,
                 optim: th.optim = th.optim.AdamW, initialize_feat: bool = False,
                 relevant_etypes: Optional[Sequence[str]] = None, relational: bool = False,
                 edge_dropout: Optional[float | Dict[str, float]] = None, sage: bool = False,
                 node_dropout: Optional[float] = None, etype_embeddings: int = 0,
                 neg_sampling_kwargs: Optional[dict] = None, cat_all: bool = False, hgt: bool = False,
                 negative_slope: float = 0.2, activation=nn.ReLU, decoder_asymm_interleave: bool = False):
        super().__init__(feat_attr=feat_attr)
        self.self_loop = self_loop
        self.reverse_edges = reverse_edges
        self.decoder_asymmetric = decoder_asymmetric
        self.decoder_asymm_interleave = decoder_asymm_interleave
        self.decoder_method = decoder_method
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.device = device
        self.dropout = dropout
        self.exposed_feat = exposed_feat
        self.pred_, self.gc_ = None, None
        self.num_heads = num_heads
        self.norm_rep = norm_rep
        self.norm_feat = norm_feat
        self.norm_all = norm_all
        self.residual = residual
        self.dropout_ = Dropout(dropout)
        self.random_state = random_state
        self.rng_ = None
        self.homogeneous = homogeneous
        self.etype_embeddings = etype_embeddings
        self.relational = relational
        self.normalizer_feat = normalizer_feat
        self.variational = variational
        self.edge_dropout = edge_dropout
        self.relevant_etypes = relevant_etypes
        self.k = k
        self.initialize_feat = initialize_feat
        self.optim = optim
        self.node_dropout = node_dropout
        self.neg_sampling_kwargs = {} if neg_sampling_kwargs is None else neg_sampling_kwargs
        self.cat_all = cat_all
        self.negative_slope = negative_slope
        self.sage = sage
        self.hgt = hgt
        self.activation = activation

    @staticmethod
    def _copy_node_feat(graph: dgl.DGLHeteroGraph, new_graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        for feat_attr in graph.ndata:
            new_graph.ndata[feat_attr] = graph.ndata[feat_attr]
        return new_graph

    def _add_reverse_edges(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        het_dict = {etype: graph.edges(etype=etype) for etype in graph.canonical_etypes}
        for etype in graph.etypes:
            stype, etype, dtype = graph.to_canonical_etype(etype)
            edges = graph.edges(etype=etype)
            het_dict[(dtype, etype + '_r', stype)] = edges[::-1]
        new_graph = dgl.heterograph(het_dict, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
        return self._copy_node_feat(graph, new_graph).to(self.device)

    def _add_self_loop(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        if len(graph.etypes) == 1:
            return dgl.add_self_loop(graph)

        het_dict = {etype: graph.edges(etype=etype) for etype in graph.canonical_etypes}
        for ntype in graph.ntypes:
            nodes = graph.nodes(ntype=ntype)
            edges = (nodes, nodes)
            het_dict[(ntype, ntype + '_self', ntype)] = edges
        new_graph = dgl.heterograph(het_dict, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
        return self._copy_node_feat(graph, new_graph).to(self.device)

    def _impute_node_feat(self, graph: dgl.DGLHeteroGraph, node_feat: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        missing_ntypes = set(graph.ntypes).difference(set(node_feat.keys()))
        if not missing_ntypes:
            return node_feat
        in_features = list(node_feat.values())[0][0].shape[-1]
        if missing_ntypes:
            if self.verbose:
                warnings.warn(
                    f'Missing {self.feat_attr} features for node types {missing_ntypes}. Initializing with zeros.')
            for ntype in missing_ntypes:
                node_feat[ntype] = th.zeros((graph.number_of_nodes(ntype=ntype), in_features)).to(self.device)
        return node_feat

    def _expose_feat(self, graph: dgl.DGLHeteroGraph, feat_names: Sequence[str],
                     as_metapath: bool = False) -> dgl.DGLHeteroGraph:
        het_dict = {etype: graph.edges(etype=etype) for etype in graph.canonical_etypes}
        num_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
        for feat_name in feat_names:
            if len(graph.ndata[feat_name].keys()) == 0:
                warnings.warn('No {} features for any nodes!'.format(feat_name))
                continue
            for ntype in graph.ntypes:
                u, v = [], []
                if ntype not in graph.ndata[feat_name]:
                    continue
                feat = graph.ndata[feat_name][ntype]
                if th.std(feat) == 0:
                    warnings.warn(f'Skipping {ntype} as it has constant {feat_name} features')
                    continue
                feat_values = th.unique(feat, dim=0)
                assert th.unique(feat_values.sum(1)) == th.tensor(1), f'Cannot expose {feat_name} as it is not one-hot'
                for feat_id, feat_value in enumerate(feat_values):
                    is_val = th.all(feat == feat_value, 1)
                    feat_nodes = graph.nodes(ntype=ntype)[is_val]
                    u.append(feat_nodes), v.append(th.full_like(feat_nodes, feat_id))

                edges = (th.hstack(u), th.hstack(v))
                het_dict[(ntype, feat_name, feat_name)] = edges
                if self.reverse_edges:
                    het_dict[(feat_name, feat_name + '_r', ntype)] = edges[::-1]
                num_nodes_dict[feat_name] = len(feat_values)

        new_graph = dgl.heterograph(het_dict, num_nodes_dict=num_nodes_dict)

        new_graph.ndata[self.feat_attr] = graph.ndata[self.feat_attr]
        if self.initialize_feat:
            for nt1, etype, nt2 in set(new_graph.canonical_etypes).difference(graph.canonical_etypes):
                if nt2 not in feat_names:
                    continue
                aggregate_fn = dgl_fn.copy_u(self.feat_attr, 'm')
                new_graph.update_all(aggregate_fn, dgl_fn.mean(msg="m", out=self.feat_attr), etype=etype)

        return new_graph.to(self.device)

    def _prep_graph(self, graph: dgl.DGLHeteroGraph, node_feat: Optional[Dict[str, np.ndarray]] = None) -> (
            dgl.DGLHeteroGraph, Dict[str, th.Tensor]):

        if self.reverse_edges:
            graph = self._add_reverse_edges(graph)
        else:
            # move to cuda for performance reasons
            graph = graph.to(self.device)
        if self.exposed_feat:
            graph = self._expose_feat(graph, self.exposed_feat)
        if self.normalizer_feat:
            graph = self._expose_feat(graph, self.normalizer_feat)
        if self.self_loop:
            graph = self._add_self_loop(graph)

        node_feat = graph.ndata[self.feat_attr] if node_feat is None else node_feat  # .copy()
        assert len(node_feat), 'need feature values for at least one node type'

        if len(graph.ntypes) > 1:
            node_feat = self._impute_node_feat(graph, node_feat)
        if self.homogeneous:
            graph = dgl.to_homogeneous(graph, ndata=[self.feat_attr])
            node_feat = th.cat(list(node_feat.values()))
        else:
            node_feat = {k: feat.to(self.device) if type(feat) == th.Tensor else [f.to(self.device) for f in feat] for
                         k, feat in node_feat.items()}

        return graph, node_feat

    def _repr(self, X: dgl.DGLHeteroGraph, h: Union[Dict[str, th.Tensor], Dict[str, np.ndarray]], ntypes=None,
              training: bool = False):
        def _dict_stack(feat: Dict[str, th.Tensor]) -> th.Tensor:
            if type(feat) is dict:
                if ntypes:
                    feat = th.vstack([feat[ntype] for ntype in ntypes])
                else:
                    feat = th.vstack(list(feat.values()))
            return feat

        if self.norm_feat:
            # REF https://arxiv.org/pdf/2108.08046v2, https://arxiv.org/pdf/2112.14936
            if type(h) is dict:
                h = {nt: th.nn.functional.normalize(h_, p=2, dim=1) for nt, h_ in h.items()}
            else:
                h = th.nn.functional.normalize(h, p=2, dim=1)

        if self.gc_:
            if self.relational:
                h = self.gc_(X, h, etypes=X.edata[dgl.ETYPE])
            elif self.hgt:
                h = self.gc_(X, h, etype=X.edata[dgl.ETYPE], ntype=X.ndata[dgl.NTYPE])
            else:
                h = self.gc_(X, h)

        if self.variational:
            self.mean_, self.log_std_ = _dict_stack(h), _dict_stack(self.std_gc_(X, h))
            gaussian_noise = th.randn(self.mean_.shape).to(self.device)
            h = self.mean_ + ((gaussian_noise * th.exp(self.log_std_).to(self.device)) if training else 0)

        h = _dict_stack(h)

        if h.ndim == 3:
            h = th.flatten(h, start_dim=1, end_dim=2)

        if self.dropout:
            h = self.dropout_(h)

        return h

    def transform(self, X: dgl.DGLHeteroGraph, nfeat: Optional[Dict[str, np.ndarray]] = None):
        with th.no_grad():
            if self.gc_:
                self.gc_.eval()
            graph, node_feat = self._prep_graph(X, nfeat)

            h = self._repr(graph, node_feat, ntypes=X.ntypes)

            if len(X.ntypes) > 1 and len(graph.ntypes) == 1:
                og_ntypes = th.Tensor([i for i, ntype in enumerate(X.ntypes)]).to(self.device)
                mask = th.isin(graph.ndata[dgl.NTYPE], og_ntypes).to(self.device)
                h = h[mask]

            return h

    def predict_proba(self, inference_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph,
                      node_feat: Optional[Dict[str, np.ndarray]] = None) -> th.Tensor:
        with th.no_grad():
            h = self.transform(inference_graph, node_feat)

            if len(h) != indicator_graph.num_nodes():
                indicator_mask = {ntype: th.isin(inference_graph.nodes(ntype=ntype), indicator_graph.nodes(ntype=ntype))
                                  for ntype in indicator_graph.ntypes}
                indicator_mask = th.hstack(list(indicator_mask.values()))
                h = h[indicator_mask]

            assert h.shape[
                       0] == indicator_graph.num_nodes(), f'Got {h.shape[0]} representations for {indicator_graph.num_nodes()} test nodes'

            self.pred_.eval()
            return self.pred_(indicator_graph.to(self.device), h).detach().cpu()

    def predict(self, inference_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph,
                node_feat: Optional[Dict[str, np.ndarray]] = None, ) -> th.Tensor:
        probas = self.predict_proba(inference_graph, indicator_graph, node_feat=node_feat)
        return th.BoolTensor(probas > .5)

    def _build_layers(self, train_graph: dgl.DGLHeteroGraph, node_feat: Dict[str, th.Tensor]) -> MixedSequential:
        in_features = list(node_feat.values())[0][0].shape[-1] if type(node_feat) is dict else node_feat.shape[-1]
        num_heads = 1 if self.num_heads is None else self.num_heads

        gc_layers = []

        if list(self.layer_sizes) in ([], None, [0], (0,)):
            self.layer_sizes = [in_features]
            return

        for i, layer_size in enumerate(self.layer_sizes):
            is_last = i == len(self.layer_sizes) - 1
            etypes = train_graph.etypes
            if len(etypes) > 1:
                gcs = {}
                for etype in etypes:
                    if self.num_heads:
                        gc = GATConv(in_features, layer_size, num_heads=self.num_heads,
                                     negative_slope=self.negative_slope)
                    elif self.sage:
                        gc = SAGEConv(in_features, layer_size, aggregator_type='mean')
                    else:
                        gc = GraphConv(in_features, layer_size)
                    gcs[etype] = gc
                gc = HeteroGraphConv(gcs)
            else:
                num_rels = len(train_graph.edata[dgl.ETYPE].unique())
                num_ntypes = len(train_graph.ndata[dgl.NTYPE].unique())
                if self.num_heads:
                    if self.etype_embeddings > 0:
                        gc = EGATConv(in_features, layer_size, num_heads=self.num_heads, num_etypes=num_rels,
                                      edge_feats=self.etype_embeddings, negative_slope=self.negative_slope)
                    elif self.hgt:
                        gc = HGTConv(in_features, layer_size // num_heads, num_heads=num_heads,
                                     num_etypes=num_rels, num_ntypes=num_ntypes)
                    else:
                        gc = GATConv(in_features, layer_size, num_heads=self.num_heads,
                                     negative_slope=self.negative_slope)
                elif self.relational:
                    gc = RelGraphConv(in_features, layer_size, num_rels=num_rels)
                elif self.sage:
                    gc = SAGEConv(in_features, layer_size, aggregator_type='mean')
                else:
                    gc = GraphConv(in_features, layer_size, allow_zero_in_degree=not self.self_loop)
            gc_layers.append(gc)
            in_features = layer_size

            if not is_last:
                gc_layers.append(self.activation())

        return MixedSequential(gc_layers, residual=self.residual, device=self.device, cat_all=self.cat_all,
                               norm=self.norm_all) if len(gc_layers) > 1 else gc_layers[0].to(self.device)

    def partial_fit(self, train_graph: dgl.DGLHeteroGraph, node_feat: Optional[Dict[str, np.ndarray]] = None,
                    train_graph_t: Optional[dgl.DGLHeteroGraph] = None, epochs: int = 1):
        def _compute_loss(h, pos_graph, neg_graph):
            num_pos_edges, num_neg_edges = pos_graph.number_of_edges(), neg_graph.number_of_edges()

            pos_probas = self.pred_(pos_graph, h)
            neg_probas = self.pred_(neg_graph, h)

            batch_loss = th.zeros(1, device=self.device)

            pos_loss = F.binary_cross_entropy(pos_probas,
                                              th.ones(pos_graph.number_of_edges(), device=self.device))
            neg_loss = F.binary_cross_entropy(neg_probas,
                                              th.zeros(neg_graph.number_of_edges(), device=self.device))
            batch_loss += pos_loss + neg_loss

            if self.variational:
                kl_divergence = 0.5 / (num_pos_edges + num_neg_edges) * (
                        1 + 2 * self.log_std_ - self.mean_ ** 2 - th.exp(self.log_std_) ** 2).sum(1).mean()
                batch_loss -= kl_divergence

            return batch_loss

        def _edge_dropout(g: dgl.DGLHeteroGraph):
            # REF https://arxiv.org/abs/1907.10903
            edges = {}
            for etype in g.etypes:
                original_edges = g.edges(etype=etype, form='eid')
                if etype not in relevant_etypes and etype.replace('_r', '') not in relevant_etypes and len(
                        g.etypes) > 1:
                    edges[etype] = original_edges
                    continue

                num_edges = int(round((1 - self.edge_dropout) * train_graph_t.num_edges(etype)))
                edges[etype] = self.rng_.choice(original_edges.cpu(), replace=False, size=num_edges)
            return dgl.edge_subgraph(g, edges, relabel_nodes=False)

        def node_dropout(g: dgl.DGLHeteroGraph):
            # REF https://arxiv.org/abs/2111.06283
            nodes = {}
            for ntype in g.ntypes:
                original_nodes = g.nodes(ntype=ntype)
                num_nodes = int(round((1 - self.node_dropout) * train_graph_t.num_nodes(ntype)))
                nodes[ntype] = self.rng_.choice(original_nodes.cpu(), replace=False, size=num_nodes)
            return dgl.node_subgraph(g, nodes, relabel_nodes=False)

        if self.gc_:
            self.gc_.train()
        self.pred_.train()
        if train_graph_t is None:
            train_graph_t, node_feat = self._prep_graph(train_graph.clone(), node_feat)

        relevant_etypes = train_graph.etypes if self.relevant_etypes is None else self.relevant_etypes
        batch_graph = dgl.edge_type_subgraph(train_graph if self.homogeneous else train_graph_t,
                                             relevant_etypes)

        for epoch in range(epochs):
            train_graph_t_ = train_graph_t
            if self.edge_dropout is not None and (type(self.edge_dropout) is dict or self.edge_dropout > 0):
                train_graph_t_ = _edge_dropout(train_graph_t_)

            if self.node_dropout is not None and self.node_dropout > 0:
                train_graph_t_ = node_dropout(train_graph_t_)

            graph_gen_time, rep_gen_time, pred_time, bwd_time = 0, 0, 0, 0
            start = time.time()

            neg_graph = util.construct_negative_graph(batch_graph, random_state=epoch, copy_feat=False, k=self.k,
                                                      verify=False, **self.neg_sampling_kwargs)
            graph_gen_time += (time.time() - start)

            # forward
            start = time.time()
            h = self._repr(train_graph_t_, node_feat, ntypes=batch_graph.ntypes, training=True)
            rep_gen_time += (time.time() - start)

            start = time.time()
            graph_gen_time += (time.time() - start)

            start = time.time()
            loss = _compute_loss(h, batch_graph, neg_graph)
            pred_time += (time.time() - start)

            # backward
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()
            bwd_time += (time.time() - start)

            if self.verbose > 2:
                print('graph_gen:', round(graph_gen_time * 1000), 'ms\trep gen:', round(rep_gen_time * 1000),
                      'ms\tpred:', round(pred_time * 1000), 'ms\tbwd:', round(bwd_time * 1000), 'ms')
            if epoch % 5 == 0 and self.verbose:
                print(f'epoch {epoch}\t loss: {round(loss.item(), 3)}')
        return self

    def fit(self, train_graph: dgl.DGLHeteroGraph, node_feat: Optional[Dict[str, np.ndarray]] = None):
        self.rng_ = np.random.RandomState(self.random_state)
        os.environ["PYTHONHASHSEED"] = str(self.random_state)
        th.cuda.empty_cache()
        th.manual_seed(self.random_state)
        th.cuda.manual_seed(self.random_state)
        th.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        th.use_deterministic_algorithms(True, warn_only=False)

        train_graph_t, node_feat = self._prep_graph(train_graph.clone(), node_feat)

        self.gc_ = self._build_layers(train_graph_t, node_feat)
        if self.variational:
            rep_size = self.layer_sizes[-1] * (self.num_heads if self.num_heads else 1)
            if self.homogeneous:
                self.std_gc_ = GATConv(rep_size, self.layer_sizes[-1],
                                       num_heads=self.num_heads) if self.num_heads else GraphConv(
                    rep_size, rep_size)
            else:
                self.std_gc_ = HeteroGraphConv(
                    {etype: GATConv(rep_size, self.layer_sizes[-1],
                                    num_heads=self.num_heads) if self.num_heads else GraphConv(
                        rep_size, rep_size) for etype in train_graph_t.etypes})

            self.std_gc_.to(self.device)
        rep_sizes = [s for s in self.layer_sizes]
        rep_size = int(np.sum(rep_sizes)) if self.cat_all else rep_sizes[-1]
        self.pred_ = LinkDecoder(rep_size, method=self.decoder_method, asymmetric=self.decoder_asymmetric,
                                 etypes=train_graph.etypes, norm=self.norm_rep,
                                 asymm_interleave=self.decoder_asymm_interleave).to(self.device)

        parameters = list(self.pred_.parameters()) if self.gc_ is None else list(self.gc_.parameters()) + list(
            self.pred_.parameters())
        if self.variational:
            parameters = parameters + list(self.std_gc_.parameters())
        self.optimizer_ = self.optim(parameters, lr=self.lr, weight_decay=self.weight_decay)

        return self.partial_fit(train_graph, node_feat, train_graph_t, epochs=self.epochs)


class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, est: [ClassifierMixin, BaseEstimator], feat_attr: str = 'feat', decoder_method: str = 'cat',
                 trans: [BaseEstimator, TransformerMixin] = None, random_state: Optional[int] = None):
        self.feat_attr = feat_attr
        self.est = est
        self.trans = trans
        self.random_state = random_state
        self.decoder_method = decoder_method

    def _decode(self, emb_u, emb_v):
        if self.decoder_method in ('cat', 'concat'):
            res = np.hstack([emb_u, emb_v])

        if self.decoder_method in ('ip', 'dot'):
            res = th.unsqueeze(th.sum(emb_u * emb_v, dim=1), dim=1)

        if self.decoder_method in ('mul', 'hadamard'):
            res = emb_u * emb_v

        if self.decoder_method == 'l1':
            res = np.abs(emb_u - emb_v)

        if self.decoder_method == 'l2':
            res = np.square(emb_u - emb_v)

        if self.decoder_method == 'avg':
            res = (emb_u + emb_v) / 2

        if self.decoder_method == 'diff':
            res = emb_v - emb_u

        return res

    def fit(self, train_graph: dgl.DGLHeteroGraph):
        train_graph_ = dgl.to_homogeneous(train_graph, ndata=[self.feat_attr])
        node_rep = train_graph_.ndata[self.feat_attr]

        if self.trans:
            node_rep = self.trans.fit_transform(node_rep)

        neg_graph = util.construct_negative_graph(train_graph, random_state=self.random_state)
        pos_e, neg_e = train_graph_.edges(), dgl.to_homogeneous(neg_graph).edges()
        edges = [*pos_e[0], *neg_e[0]], [*pos_e[1], *neg_e[1]]
        # forward
        labels = np.hstack([np.ones_like(pos_e[0]), np.zeros_like(neg_e[0])])
        edge_rep = self._decode(node_rep[edges[0]], node_rep[edges[1]])
        self.est.fit(edge_rep, labels)
        return self

    def predict_proba(self, inference_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph) -> np.ndarray:
        node_rep = dgl.to_homogeneous(inference_graph, ndata=[self.feat_attr]).ndata[self.feat_attr]
        if self.trans:
            node_rep = self.trans.transform(node_rep)

        edges = dgl.to_homogeneous(indicator_graph).edges()
        edge_rep = self._decode(node_rep[edges[0]], node_rep[edges[1]])
        return self.est.predict_proba(edge_rep)[:, 1]

    def predict(self, inference_graph: dgl.DGLHeteroGraph, indicator_graph: dgl.DGLHeteroGraph) -> np.ndarray:
        probas = self.predict_proba(inference_graph, indicator_graph)
        return probas > .5
