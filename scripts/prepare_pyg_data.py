import csv
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from parse_graphs import read_graph


def read_args(text):
    return {x.strip() for x in (text or "").split(",") if x.strip()}


def read_accepted(path):
    return read_args(path.read_text(encoding="utf-8").replace("\n", ","))


def basic_features(nodes, source, target):
    return torch.tensor(
        [[node == source, node == target] for node in nodes],
        dtype=torch.float,
    )


# compute structural node features
def structural_features(G, nodes, source, target, target_scc):
    a_in = set(G.predecessors(source))
    a_out = set(G.successors(source))
    b_in = set(G.predecessors(target))
    b_out = set(G.successors(target))

    return torch.tensor(
        [[
            node == source,
            node == target,
            node in a_in,
            node in a_out,
            node in b_in,
            node in b_out,
            node in target_scc,
        ] for node in nodes],
        dtype=torch.float,
    )


# build dataset
def build_data(G, row, feature_set, accepted, target_scc, reach):
    nodes = list(G)
    source, target = row["source"], row["target"]

    if feature_set in {"basic", "basic_cad"}:
        x = basic_features(nodes, source, target)
    elif feature_set in {"struct", "struct_cad"}:
        x = structural_features(G, nodes, source, target, target_scc)
    else:
        raise ValueError("feature_set must be basic, basic_cad, struct, or struct_cad")

    if feature_set in {"basic_cad", "struct_cad"}:
        x = torch.cat([
            x,
            torch.tensor([[node in accepted] for node in nodes], dtype=torch.float),
        ], dim=1)

    delta = read_args(row["delta_args"])
    gain = read_args(row["gain_args"])
    loss = read_args(row["loss_args"])
    idx = {node: i for i, node in enumerate(nodes)}

    data = from_networkx(G)
    data.edge_index = torch.cat([data.edge_index, torch.tensor([[idx[source]], [idx[target]]], dtype=torch.long),], dim=1)
    data.edge_attr = torch.zeros((data.edge_index.size(1), 1), dtype=torch.float)
    data.edge_attr[-1, 0] = 1.0

    excluded = (
            set(G.predecessors(source))
            | set(G.successors(source))
            | set(G.predecessors(target))
    )

    data.x = x
    data.y = torch.tensor([node in delta for node in nodes], dtype=torch.float)
    data.y_dir = torch.tensor([
        0 if node in gain else 1 if node in loss else -1
        for node in nodes
    ], dtype=torch.long)
    data.c_mask = torch.tensor(
        [node in reach - excluded for node in nodes],
        dtype=torch.bool,
    )

    data.node_names = nodes
    data.source = source
    data.target = target
    return data


def load_split(dataset_dir, split, feature_set):
    split_dir = Path(dataset_dir) / split
    graphs, cad, scc_of, reach = {}, {}, {}, {}
    data_list = []

    with (split_dir / "instances.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["file_name"]

            if name not in graphs:
                G = read_graph(split_dir / "base_graphs" / name)
                graphs[name] = G
                reach[name] = {}
                if feature_set in {"struct", "struct_cad"}:
                    scc_of[name] = {
                        node: component
                        for component in nx.strongly_connected_components(G)
                        for node in component
                    }

            G = graphs[name]
            target = row["target"]
            if target not in reach[name]:
                reach[name][target] = nx.descendants(G, target) | {target}

            accepted = None
            if feature_set in {"basic_cad", "struct_cad"}:
                if name not in cad:
                    cad[name] = read_accepted(
                        split_dir / "base_cad" / f"{name}.dca.txt"
                    )
                accepted = cad[name]

            data = build_data(
                G,
                row,
                feature_set,
                accepted,
                scc_of[name][target] if name in scc_of else None,
                reach[name][target],
            )
            data.graph_id = row["graph_id"]
            data.file_name = name
            data_list.append(data)

    return data_list


def make_loader(dataset_dir, split, feature_set, batch_size, shuffle=False):
    return DataLoader(
        load_split(dataset_dir, split, feature_set),
        batch_size=batch_size,
        shuffle=shuffle,
    )
