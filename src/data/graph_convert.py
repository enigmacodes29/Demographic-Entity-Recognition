"""
NetworkX DER graph -> PyTorch Geometric HeteroData, for GAT/R-GCN/RGAT.

Extracted verbatim from Copy_of_Graph_models.ipynb, after confirming byte-for-byte identity
with the RGAT notebook's version. The Sentence node is deliberately excluded -- only annotated
DER nodes and DER-to-DER relations reach the graph encoders. Depends on der_graph.py for
SENTENCE_NODE_TYPE and relation_family.
"""
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

from .der_graph import SENTENCE_NODE_TYPE, relation_family


def nx_to_hetero(G, label, feat, collapse_relations=False, zero_sentence=False):
    """Convert a NetworkX sentence graph to a DER-only HeteroData graph.

    The Sentence node is deliberately excluded. Only annotated DER nodes, their text-only
    features, and relations between DER nodes reach the graph encoder. Edges touching the
    Sentence node are discarded.
    """
    data = HeteroData()
    node_maps = {}
    der_nodes = [(n, a) for n, a in G.nodes(data=True)
                if a.get("node_type", "Unknown") != SENTENCE_NODE_TYPE]
    node_types = sorted({a.get("node_type", "Unknown") for _, a in der_nodes})

    for nt in node_types:
        nodes = [n for n, a in der_nodes if a.get("node_type", "Unknown") == nt]
        node_maps[nt] = {n: i for i, n in enumerate(nodes)}
        feat.ensure([feat.node_key(G, n) for n in nodes])
        data[nt].x = torch.stack([feat.get(G, n) for n in nodes]).float()

    edge_dict = defaultdict(list)
    for s, t, a in G.edges(data=True):
        st = G.nodes[s].get("node_type", "Unknown")
        tt = G.nodes[t].get("node_type", "Unknown")
        if st == SENTENCE_NODE_TYPE or tt == SENTENCE_NODE_TYPE:
            continue
        if st not in node_maps or tt not in node_maps:
            continue
        rel = a.get("relation", "unknown")
        if collapse_relations:
            rel = relation_family(rel)
        si, ti = node_maps[st][s], node_maps[tt][t]
        edge_dict[(st, rel, tt)].append([si, ti])
        edge_dict[(tt, "rev_" + rel, st)].append([ti, si])

    for et, e in edge_dict.items():
        data[et].edge_index = torch.tensor(e, dtype=torch.long).t().contiguous()
    data.y = torch.tensor([int(label)], dtype=torch.long)
    return data


def convert(graphs_nx, labels, feat, **kw):
    return [nx_to_hetero(G, y, feat, **kw) for G, y in zip(graphs_nx, labels)]


def build_metadata(data_list):
    nts, ets = set(), set()
    for g in data_list:
        n, e = g.metadata()
        nts.update(n)
        ets.update(e)
    return sorted(nts), sorted(ets)
