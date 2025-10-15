from typing import Dict, Iterator
from typing import Sequence, Optional, Union

import dgl
import dgl.function as fn
import torch as th
import torch.nn.functional as F
from dgl import DGLError
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import Parameter, Identity


class MixedSequential(nn.Module):
    def __init__(self, modules: Sequence[nn.Module], residual: bool = False, device: str = 'cpu', cat_all: bool = False,
                 norm: bool = False):
        super().__init__()
        self.modules = [m.to(device) for m in modules]
        self.residual = residual
        self.device = device
        self.cat_all = cat_all
        self.norm = norm

    def forward(self, g: dgl.DGLHeteroGraph, h: Union[th.Tensor, Dict[str, th.Tensor]], **kwargs) -> Union[
        th.Tensor, Dict[str, th.Tensor]]:
        res_rep = None
        h_cat = None
        for module in self.modules:
            num_kwargs = 0 if module.forward.__defaults__ is None else len(module.forward.__defaults__)
            num_args = module.forward.__code__.co_argcount - num_kwargs
            if num_args > 2:
                h = module(g, h, **kwargs)
            else:
                if type(h) is dict:
                    h = {nt: module(rep) for nt, rep in h.items()}
                else:
                    h = module(h)

            rep_shape = list(h.values())[0].shape[1:] if type(h) is dict else h.shape[1:]
            if len(rep_shape) > 1:
                if type(h) is dict:
                    h = {nt: th.mean(rep, 1) for nt, rep in h.items()}
                else:
                    h = th.mean(h, 1)

            if len(list(module.parameters())):
                if self.residual:
                    if res_rep is not None:
                        res_rep_shape = list(res_rep.values())[0].shape[1:] if type(res_rep) is dict else res_rep.shape[
                                                                                                          1:]
                        if res_rep_shape == rep_shape:
                            if type(h) is dict:
                                h = {k: v + res_rep[k] for k, v in h.items()}
                            else:
                                h += res_rep

                if self.norm:
                    if type(h) is dict:
                        h = {nt: th.nn.functional.normalize(h_, p=2, dim=1) for nt, h_ in h.items()}
                    else:
                        h = th.nn.functional.normalize(h, p=2, dim=1)

                if self.cat_all:
                    h_cat = h if h_cat is None else {nt: th.hstack([h_, h_cat[nt]]) for nt, h_ in h.items()} if type(
                        h) is dict else th.hstack([h, h_cat])
            res_rep = h
        return h_cat if self.cat_all else h

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for module in self.modules:
            yield from module.parameters(recurse)

    def __repr__(self):
        return 'MixedSequential(\n\t' + '\n\t'.join([str(m) for m in self.modules]) + '\n)'





class SemanticAttention(nn.Module):
    def __init__(self, in_size: int, hidden_size: Optional[int] = 128):
        super(SemanticAttention, self).__init__()
        if hidden_size is None or hidden_size < 1:
            self.project = nn.Linear(in_size, 1, bias=False)
        else:
            self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(),
                                         nn.Linear(hidden_size, 1, bias=False))

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = th.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)


class DistMul(nn.Module):
    def __init__(self, in_features: int, asymmetric: bool = False, activation=F.sigmoid, disjoint: bool = False,
                 etypes: Optional[Sequence[str]] = None, hidden_channels: int = 384):
        super().__init__()
        self.asymmetric = asymmetric
        self.activation = activation
        self.disjoint = disjoint
        self.rel_weight = th.nn.Parameter(th.Tensor(len(etypes), hidden_channels))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.rel_weight, gain=gain)

        if self.asymmetric:
            assert in_features % 2 == 0, f'asymmetric decoder requires even number of features, got {in_features}'

    def forward(self, g, h):
        with g.local_scope():
            u, v = dgl.to_homogeneous(g).edges()
            emb_u, emb_v = h[u], h[v]

        if self.asymmetric:
            cutoff = int(round(h.shape[1] / 2))
            emb_u, emb_v = emb_u[:, :cutoff], emb_v[:, cutoff:]

        etype = dgl.to_homogeneous(g).edata[dgl.ETYPE]
        res = th.unsqueeze(th.sum(emb_u * self.rel_weight[etype] * emb_v, dim=1), dim=1)

        if self.activation is not None:
            res = self.activation(res)

        return res.squeeze()


class LinkDecoder(nn.Module):
    def __init__(self, in_features: int, method='ip', asymmetric: bool = False, activation=F.sigmoid,
                 disjoint: bool = False, etypes: Optional[Sequence[str]] = None,  norm: bool = False,
                 asymm_interleave: bool = True):
        super().__init__()
        self.method = method
        self.asymmetric = asymmetric
        self.activation = activation
        self.disjoint = disjoint
        self.norm = norm
        self.asymm_interleave = asymm_interleave

        if self.method not in ('ip', 'dot'):
            if self.asymmetric:
                assert in_features % 2 == 0, f'asymmetric decoder requires even number of features, got {in_features}'
                in_features = int(round(in_features / 2))

            if self.method in ('cat', 'concat'):
                in_features *= 2


    def forward(self, g: dgl.DGLHeteroGraph, h, return_dict: bool = False):
        res = None
        with g.local_scope():
            u, v = dgl.to_homogeneous(g).edges()
            emb_u, emb_v = h[u], h[v]

        if self.asymmetric:
            if self.asymm_interleave:
                emb_u, emb_v = emb_u[:, ::2], emb_v[:, 1::2]
            else:
                cutoff = emb_u.shape[1] // 2
                emb_u, emb_v = emb_u[:, :cutoff], emb_v[:, cutoff:]

        if self.norm:
            emb_u = th.nn.functional.normalize(emb_u, p=2, dim=1)
            emb_v = th.nn.functional.normalize(emb_v, p=2, dim=1)

        if self.method in ('cat', 'concat'):
            res = th.hstack([emb_u, emb_v])

        if self.method in ('ip', 'dot'):
            res = th.unsqueeze(th.sum(emb_u * emb_v, dim=1), dim=1)

        if self.method in ('mul', 'hadamard'):
            res = emb_u * emb_v

        if self.method == 'l1':
            res = th.abs(emb_u - emb_v)

        if self.method == 'l2':
            res = th.square(emb_u - emb_v)

        if self.method == 'avg':
            res = (emb_u + emb_v) / 2

        if self.method not in ('ip', 'dot'):
            if self.disjoint:
                etypes = dgl.to_homogeneous(g).edata[dgl.ETYPE]
                res = [res[etypes == i] for i, etype in enumerate(g.etypes)]

        if self.activation is not None:
            res = self.activation(res)

        res = res.squeeze()
        return {et: res[dgl.to_homogeneous(g).edata[dgl.ETYPE] == i] for i, et in
                enumerate(g.etypes)} if return_dict else res


class EGATConv(nn.Module):
    """
    Adapted from
    https://github.com/THUDM/HGB/blob/master/LP/benchmark/methods/baseline/conv.py
    """

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, edge_feats: int, num_etypes: int = 0,
                 feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False, activation=None,
                 allow_zero_in_degree=False, bias=False, alpha=0., etype_arg: str = dgl.ETYPE):
        super(EGATConv, self).__init__()
        self.etype_arg = etype_arg
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            edge_types = graph.edata[self.etype_arg].int().squeeze()
            e_feat = self.edge_emb(edge_types)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst
