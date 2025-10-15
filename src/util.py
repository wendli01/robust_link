import copy
import time
import warnings
from collections import defaultdict
from typing import Optional, Sequence, Dict, Tuple, Union

import dgl
import numpy as np
import pandas as pd
import torch as th
from dgl.dataloading.negative_sampler import Uniform, GlobalUniform
from matplotlib import pyplot as plt
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def hetero_node_subgraph(graph: dgl.DGLHeteroGraph, nodes: Dict[str, Sequence[bool]]) -> dgl.DGLHeteroGraph:
    graph = graph.clone()

    if dgl.NID not in graph.ndata or len(graph.ndata[dgl.NID]) == 0:
        graph.ndata[dgl.NID] = {ntype: th.arange(graph.num_nodes(ntype=ntype)) for ntype in graph.ntypes}
    for ntype in graph.ntypes:
        remove_nodes = graph.nodes(ntype=ntype)[~nodes[ntype]]
        graph = dgl.remove_nodes(graph, remove_nodes, ntype=ntype)
        assert graph.num_nodes(ntype=ntype) == nodes[
            ntype].sum(), f"Expected {nodes[ntype].sum()} nodes, got {graph.num_nodes(ntype=ntype)}"

    return graph


def structured_negative_sampling(edges: Tuple[th.Tensor, th.Tensor], num_nodes: int):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    REF https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/utils/negative_sampling.html
    """
    device = edges[0].device
    src, dst = edges[0].cpu(), edges[1].cpu()
    pos_idx = src * num_nodes + dst

    rand = th.randint(num_nodes, (src.size(0),), dtype=th.long)
    neg_idx = src * num_nodes + rand

    mask = th.from_numpy(np.isin(neg_idx.cpu(), pos_idx.cpu())).to(th.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = th.randint(num_nodes, (rest.size(0),), dtype=th.long)
        rand[rest] = tmp
        neg_idx = src[rest] * num_nodes + tmp

        mask = th.from_numpy(np.isin(neg_idx, pos_idx)).to(th.bool)
        rest = rest[mask]

    return src.to(device), rand.to(device)



def construct_negative_graph(graph: dgl.DGLHeteroGraph, k: int = 1, etypes: Optional[Sequence[str]] = None,
                             random_state=None, copy_feat: bool = True, add_reverse: bool = False,
                             structured: bool = True, true_negative: bool = True, verify: bool = True):
    if etypes is None:
        etypes = graph.etypes
    etypes = [graph.to_canonical_etype(et) for et in etypes]

    edges = {etype: graph.edges(etype=etype, form='eid') for etype in etypes}
    if random_state:
        th.manual_seed(random_state)

    if not true_negative:
        sampler = Uniform(k=k) if structured else GlobalUniform(k=k)
        neg_edges = sampler(graph, edges)
    else:
        if structured:
            neg_edges = {
                et: structured_negative_sampling(graph.edges(etype=et), num_nodes=graph.num_nodes(ntype=et[-1])) for
                et in etypes}
        else:
            neg_edges = {
                et: dgl.sampling.global_uniform_negative_sampling(graph, num_samples=int(graph.num_edges(etype=et) * k),
                                                                  etype=et, replace=False) for et in
                etypes}

    if add_reverse:
        for etype in etypes:
            if etype[0] == etype[-1]:
                u_, v_ = graph.edges(etype=etype)[::-1]
                u, v = neg_edges[etype]
                neg_edges[etype] = (th.cat([u, u_]), th.cat([v, v_]))

    neg_g = dgl.heterograph(neg_edges, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

    if verify:
        if structured:
            for et in etypes:
                assert th.equal(graph.out_degrees(etype=et),
                                neg_g.out_degrees(etype=et)), f'out degrees do not match for {et}'

        for et in etypes:
            num_false_neg = graph.has_edges_between(*neg_g.edges(etype=et), et).sum()
            if num_false_neg > 0:
                assert not true_negative
                warnings.warn(f'{num_false_neg} {et} false negatives')

    if copy_feat:
        for feat in graph.ndata.keys():
            neg_g.ndata[feat] = graph.ndata[feat]
    return neg_g


def mrr(edge_list, confidence, labels):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param labels: shape(edge_num,)
    :return: dict with all scores we need
    """
    # TODO verify
    confidence = np.array(confidence)
    labels = np.array(labels)
    mrr_list, cur_mrr = [], 0
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(confidence[i])
    for h_id in t_dict.keys():
        conf_array = np.array(conf_dict[h_id])
        rank = np.argsort(-conf_array)
        sorted_label_array = np.array(labels_dict[h_id])[rank]
        pos_index = np.where(sorted_label_array == 1)[0]
        if len(pos_index) == 0:
            continue
        pos_min_rank = np.min(pos_index)
        cur_mrr = 1 / (1 + pos_min_rank)
        mrr_list.append(cur_mrr)

    return np.mean(mrr_list)


def _inject_fake_edges(g: dgl.DGLHeteroGraph, noise_ratio: float, verbose: bool = False,
                       etypes: Sequence[str] = None) -> dgl.DGLHeteroGraph:
    g_ = copy.deepcopy(g)
    for et in g.etypes if etypes is None else etypes:
        neg_edges = dgl.sampling.global_uniform_negative_sampling(g, etype=et, replace=False,
                                                                  num_samples=int(g.num_edges(etype=et) * noise_ratio))

        g_ = dgl.add_edges(g_, *neg_edges, etype=et)
        if verbose > 0:
            print(f"Injected {g_.num_edges(etype=et) - g.num_edges(etype=et)} fake edges into edge type {et}.")
    return g_


def date_score_lp(estimator, graph: dgl.DGLHeteroGraph, dates: Dict[str, pd.Series], n_splits: int = 5,
                  verbose: bool = False, k: int = 1, test_size: Union[float, Sequence[float]] = .9, random_state=None,
                  plot_roc: bool = False, scoring: Tuple = (average_precision_score, roc_auc_score),
                  test_splits: int = 5, etypes: Optional[Sequence[str]] = ('CC', 'CL'), cumulative_testing: bool = True,
                  cumulative_training: bool = True, fully_inductive: bool = False, noise_ratio: float = 0,
                  **negative_sampling_kwargs) -> pd.DataFrame:
    def date_split_nodes(node_dates: pd.Series, cutoff: float):
        num_nodes_req = cutoff * len(node_dates)
        num_nodes = pd.Series(node_dates).value_counts().sort_index().cumsum()
        split_date = num_nodes.index[np.where(num_nodes >= num_nodes_req)[0][0]]

        if cumulative_training:
            train_node_ids = node_dates.values < split_date
        else:
            prev_cutoff = (cutoff - 1 / n_splits) * len(node_dates)
            prev_split_date = num_nodes.index[np.where(num_nodes >= prev_cutoff)[0][0]]
            train_node_ids = (node_dates.values >= prev_split_date) & (node_dates.values < split_date)

        if cumulative_testing:
            test_node_ids = node_dates.values >= split_date
        else:
            next_cutoff = (cutoff + 1 / n_splits) * len(node_dates)
            next_split_date = num_nodes.index[np.where(num_nodes >= next_cutoff)[0][0]]
            test_node_ids = (node_dates.values >= split_date) & (node_dates.values < next_split_date)
        if verbose > 1:
            print('cutoff date', split_date, 'training nodes:', train_node_ids.sum(), 'test nodes:',
                  test_node_ids.sum())
        return train_node_ids, test_node_ids

    def get_split_graphs() -> (dgl.DGLHeteroGraph, dgl.DGLHeteroGraph):
        def get_new_edges(etype):
            nt1, etype, nt2 = graph.to_canonical_etype(etype)
            edges = []
            if nt1 in test_nodes:
                out_edges = graph.out_edges(graph.nodes(ntype=nt1)[test_nodes[nt1]], etype=etype, form='eid')
                edges.append(out_edges)
            if nt2 in test_nodes:
                in_edges = graph.in_edges(graph.nodes(ntype=nt2)[test_nodes[nt2]], etype=etype, form='eid')
                edges.append(in_edges)

            return th.unique(th.hstack(edges)).tolist()

        node_split = {nt: date_split_nodes(nd.fillna(''), cutoff=cutoff) for nt, nd in dates.items()}

        train_nodes = {
            nt: node_split[nt][0] if nt in node_split else th.ones(graph.num_nodes(ntype=nt), dtype=th.bool) for
            nt in graph.ntypes}

        train_graph = hetero_node_subgraph(graph, nodes=train_nodes)

        test_nodes = {
            nt: node_split[nt][1] if nt in node_split else th.zeros(graph.num_nodes(ntype=nt), dtype=th.bool) for
            nt in graph.ntypes}
        new_edges = {etype: get_new_edges(etype) for etype in etypes}
        old_edges = {et: list(set(graph.edges(etype=et, form='eid').tolist()).difference(new_edges[et])) for et in
                     etypes}

        for etype in etypes:
            if cumulative_testing and cumulative_training:
                assert len(new_edges[etype]) + train_graph.number_of_edges(etype) == graph.number_of_edges(
                    etype), 'lost edges for ' + etype

        return train_graph, new_edges, old_edges

    def score_fold(estimator: BaseEstimator, random_state=None, node_feat=None) -> (
            pd.DataFrame, Sequence, Sequence):
        def get_test_data(g: dgl.DGLHeteroGraph, test_size_: float) -> (
                dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, dgl.DGLHeteroGraph, th.Tensor):
            test_edges = {
                et: rng.choice(new_edges[et], size=int(test_size_ * len(new_edges[et])), replace=False).tolist()
                for et in etypes}

            inference_edges = {}
            for et in etypes:
                inference_edges[et] = list(set(new_edges[et]).difference(test_edges[et]))
                if not fully_inductive:
                    inference_edges[et] += list(old_edges[et])

            test_graph_pos = dgl.edge_subgraph(g, test_edges, relabel_nodes=False)
            inference_graph = dgl.edge_subgraph(g, inference_edges, relabel_nodes=False)

            assert not any(
                [any(inference_graph.has_edges_between(*test_graph_pos.edges(etype=et), etype=et)) for et in etypes])

            test_graph_neg = construct_negative_graph(test_graph_pos, k=k, random_state=random_state,
                                                      **negative_sampling_kwargs)
            if verbose > 2:
                print('train graph', train_graph, '\ninference graph', inference_graph, '\ntest graph pos',
                      test_graph_pos, '\ntest graph neg', test_graph_neg)
            return inference_graph, test_graph_pos, test_graph_neg

        def score_etype(scorer, etype: str) -> Dict[str, float]:
            def _get_etype_edges(g):
                etype_id = g.etypes.index(etype)
                return dgl.to_homogeneous(g).edata[dgl.ETYPE] == etype_id

            is_etype = np.hstack([_get_etype_edges(test_graph_pos), _get_etype_edges(test_graph_neg)])
            y_, probas_ = y[is_etype], probas[is_etype]
            return scorer(y_, probas_)

        def score_split(y, probas) -> Union[
            Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
            scores = {}

            for scorer in scoring:
                scorer_name = scorer.__name__.replace('_score', '').replace('average_precision', 'ap')
                scores['micro_' + scorer_name] = scorer(y, probas)
                results = []
                if len(etypes) > 1:
                    for etype in etypes:
                        results.append(score_etype(scorer, etype))
                        scores[etype + '_' + scorer_name] = score_etype(scorer, etype)
                    scores['macro_' + scorer_name] = np.mean(results)

            test_edges = np.hstack(
                [dgl.to_homogeneous(test_graph_pos).edges(), dgl.to_homogeneous(test_graph_neg).edges()])
            scores['mrr'] = mrr(test_edges, probas, y)

            fpr, tpr, _ = roc_curve(y, probas)
            return scores, fpr, tpr

        train_start = time.time()
        estimator = clone(estimator).fit(train_graph)
        train_time = time.time() - train_start

        scores, fprs, tprs = [], [], []
        for test_size_ in [test_size] if type(test_size) is float else test_size:
            for _ in range(test_splits):
                inference_graph, test_graph_pos, test_graph_neg = get_test_data(g=graph, test_size_=test_size_)

                test_start = time.time()
                pos_probas = estimator.predict_proba(inference_graph, test_graph_pos)
                neg_probas = estimator.predict_proba(inference_graph, test_graph_neg)
                test_time = time.time() - test_start

                y = np.hstack([th.ones(test_graph_pos.number_of_edges()), th.zeros(test_graph_neg.number_of_edges())])
                probas = np.hstack([pos_probas, neg_probas])

                split_score, fpr, tpr = score_split(y, probas)
                split_score['test_time'] = test_time

                if type(test_size) != float:
                    split_score['test_size'] = test_size_
                scores.append(split_score)
                fprs.append(fpr)
                tprs.append(tpr)

            if verbose:
                print(pd.DataFrame(scores[-test_splits:]).agg(['mean', 'std']).round(3))

        scores = pd.DataFrame(scores)
        scores['train_time'] = train_time

        return scores, fprs, tprs

    rng = np.random.RandomState(random_state)
    dgl.seed(random_state)
    scores = []

    if etypes is None:
        etypes = graph.etypes

    if noise_ratio > 0:
        warnings.warn('Injecting fake edges as a robustness check!')

    for cutoff in np.arange(0, 1, 1 / n_splits)[1:]:
        train_graph, new_edges, old_edges = get_split_graphs()

        if noise_ratio > 0:
            train_graph = _inject_fake_edges(train_graph, noise_ratio=noise_ratio, verbose=verbose - 1, etypes=etypes)

        score_df, fprs, tprs = score_fold(estimator=estimator)
        score_df['cutoff'] = cutoff

        if plot_roc:
            for fpr, tpr in zip(fprs, tprs):
                plt.plot(fpr, tpr, label=f'ROC fold {len(scores)}')
        scores.append(score_df)

    return pd.concat(scores)
