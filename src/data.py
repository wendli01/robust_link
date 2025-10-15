import pickle
from typing import Tuple, Optional

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from neo4j import GraphDatabase
from sklearn.preprocessing import OneHotEncoder

GERMAN_STATES = (
    'Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern',
    'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
    'Schleswig-Holstein', 'Thüringen', 'NRW', 'BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL',
    'SN', 'ST', 'SH', 'TH')

REMOVE_STATES = {'|'.join(GERMAN_STATES): '[STA]'}


def from_neo4j(host: str = 'bolt://localhost:7687', auth=("neo4j", "neo4j"),
               label_mapping: dict = (('Legislation', 'Law'),)) -> nx.MultiDiGraph:
    with GraphDatabase.driver(host, auth=auth).session() as session:
        node_records = session.run("MATCH (n) RETURN n")
        nodes = list(node_records.graph()._nodes.values())

    G = nx.MultiDiGraph()

    label_mapping = {on: nn for on, nn in label_mapping}

    for node in nodes:
        properties = {**{k: str(v) for k, v in node._properties.items()}}
        label = None if len(node._labels) < 1 else list(node._labels)[0]
        properties['labels'] = label if label not in label_mapping else label_mapping[label]
        G.add_node(node.id, **properties)

    with GraphDatabase.driver(host, auth=auth).session() as session:
        rel_records = session.run("MATCH ()-[r]->() RETURN r")
        rels = list(rel_records.graph()._relationships.values())

    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

    return G


def load_old_graph(nx_dump_fp: Optional[str] = 'data/old/old_ref.pkl', max_expression_ratio: float = .5) -> Tuple[
    dgl.DGLHeteroGraph, pd.DataFrame, pd.DataFrame]:
    def construct_data_df(ntype: str):
        return pd.DataFrame([G.nodes[n] for n in type_nodes[ntype]]).drop(columns=['labels'])

    with open(nx_dump_fp, 'rb') as fb:
        G = pickle.load(fb)

    node_types = nx.get_node_attributes(G, 'labels')
    node_types = {n: node_types[n] for n in sorted(node_types)}

    node_counts = pd.Series(list(node_types.values())).value_counts()
    ntypes = set(node_counts.index)
    assert ntypes == {'Case', 'Court', 'Law'}, 'Graph needs Case, Court and Law nodes'

    type_nodes = {nt: [n for n, nt_ in node_types.items() if nt == nt_] for nt in ntypes}
    num_nodes_dict = {nt: node_counts[nt] for nt in ntypes}

    node_id_mapping = {nt: {old_i: new_i for new_i, old_i in enumerate(type_nodes[nt])} for nt in ntypes}
    het_dict = {}
    for u, v in list(G.edges()):
        nt1, nt2 = node_types[u], node_types[v]
        if nt1 not in ntypes or nt2 not in ntypes:
            continue

        if (nt1, nt2) == ('Court', 'Case'):
            nt2, nt1 = (nt1, nt2)
            u, v = (v, u)

        et = nt1[0] + nt2[0]
        if (nt1, et, nt2) not in het_dict:
            het_dict[(nt1, et, nt2)] = []
        e = (node_id_mapping[nt1][u], node_id_mapping[nt2][v])
        het_dict[(nt1, et, nt2)].append(e)

    refs_g = dgl.heterograph(het_dict, num_nodes_dict=num_nodes_dict)

    case_df, law_df, court_df = construct_data_df('Case'), construct_data_df('Law'), construct_data_df('Court')
    case_court_df = court_df.iloc[refs_g.edges(etype=('Case', 'CC', 'Court'))[1]]
    case_df = pd.concat([case_df, case_court_df.add_prefix('court_').reset_index(drop=True)], axis=1)

    ndata = {}

    for ntype, df in zip(['Case', 'Law'], [case_df, law_df]):
        for feat_name in df.columns:
            feat = df[feat_name]
            feat_name = str.lower(ntype + '_' + feat_name)

            if feat_name.endswith('date'):
                feat = pd.to_datetime(feat).values.astype(np.int64) // 10 ** 11
            elif feat.dtype == 'object' or (feat.dtype == np.int64 and not feat_name.endswith('id')):
                if len(feat.unique()) >= max_expression_ratio * refs_g.num_nodes(ntype):
                    continue
                feat = OneHotEncoder().fit_transform(feat.values.reshape(-1, 1)).toarray()

            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)

            if feat_name not in ndata:
                ndata[feat_name] = {}
            ndata[feat_name][ntype] = th.FloatTensor(feat)

    refs_g.ndata.update(ndata)

    return refs_g.node_type_subgraph(['Case', 'Law']), case_df, law_df
