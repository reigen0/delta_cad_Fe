"""Shared pre-update description for candidate generation and exhaustive analysis.

Every neighborhood and reachability set refers to F, not F+e.
Integer bit masks keep candidate enumeration inexpensive; no solver is called.
"""
import hashlib
import networkx as nx


def make_mask(values, index):
    mask = 0
    for x in values:
        mask |= 1 << index[x]
    return mask


def mask_arguments(mask, cache):
    """Decode a region"""
    values = set()
    while mask:
        bit = mask & -mask
        values.add(cache["ordered_nodes"][bit.bit_length() - 1])
        mask ^= bit
    return values


def graph_cache(graph, accepted):
    nodes = sorted(graph.nodes())
    index = {x: i for i, x in enumerate(nodes)}
    sccs = list(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, sccs)
    mapping = condensation.graph["mapping"]
    return {
        "ordered_nodes": nodes, "nodes": set(nodes), "index": index,
        "all": (1 << len(nodes)) - 1, "cad": make_mask(accepted, index),
        "pred": {x: make_mask(graph.predecessors(x), index) for x in nodes},
        "succ": {x: make_mask(graph.successors(x), index) for x in nodes},
        "reach": {x: make_mask({x} | nx.descendants(graph, x), index) for x in nodes},
        "target_scc_size": {x: len(sccs[mapping[x]]) for x in nodes},
        "target_scc_in_degree": {x: condensation.in_degree(mapping[x]) for x in nodes},
        "target_scc_out_degree": {x: condensation.out_degree(mapping[x]) for x in nodes},
    }


def describe_attack(source, target, cache):
    """Return scalar descriptors, candidate-region masks, and the reachability mask.
'unresolved' means neither change exclusion nor change guarantee was established.
It is an update classification, not the argument-labelling status 'undec'.
"""
    pred, succ = cache["pred"], cache["succ"]
    a_bit, b_bit = 1 << cache["index"][source], 1 << cache["index"][target]
    all_args, cad, reach = cache["all"], cache["cad"], cache["reach"][target]
    excluded = pred[source] | succ[source] | pred[target]
    gain_C = reach & ~(excluded | b_bit) & all_args
    loss_C = reach & ~(excluded | succ[target]) & all_args
    change_C = reach & ~excluded & all_args
    gain_cad, loss_cad = gain_C & ~cad & all_args, loss_C & cad
    source_acc, target_acc = bool(cad & a_bit), bool(cad & b_bit)
    self_add, reciprocal = source == target, bool(succ[target] & a_bit)
    source_self, source_unatt = bool(succ[source] & a_bit), pred[source] == 0
    no_gain_neighbourhood = not (succ[target] & ~(b_bit | excluded) & all_args)
    gain_ruled_out = self_add or source_self or no_gain_neighbourhood or gain_cad == 0
    loss_ruled_out = not target_acc or reciprocal or loss_cad == 0
    guaranteed = (self_add and source_acc) or (source_unatt and target_acc)
    ruled_out = gain_ruled_out and loss_ruled_out
    theory = "guaranteed" if guaranteed else "ruled_out" if ruled_out else "unresolved"
    direction = ("both_ruled_out" if ruled_out else
                 "gain_ruled_out" if gain_ruled_out else
                 "loss_ruled_out" if loss_ruled_out else "neither_ruled_out")
    fields = {
        "source": source, "target": target, "n_arg": len(cache["ordered_nodes"]),
        "source_acc_F": source_acc, "target_acc_F": target_acc,
        "endpoint_status": f"{int(source_acc)}{int(target_acc)}",
        "self_attack_addition": self_add, "reciprocal": reciprocal,
        "source_self_att": source_self, "source_unatt": source_unatt,
        "cycle_closing": bool(reach & a_bit),
        "no_gain_neighborhood_condition": no_gain_neighbourhood,
        "gain_ruled_out": gain_ruled_out, "loss_ruled_out": loss_ruled_out,
        "change_guaranteed": guaranteed, "change_ruled_out": ruled_out,
        "theory_class": theory, "direction_class": direction,
        "sampling_class": f"unresolved_{direction}" if theory == "unresolved" else theory,
        "target_in_degree_ns": (pred[target] & ~b_bit).bit_count(),
        "target_out_degree_ns": (succ[target] & ~b_bit).bit_count(),
        "target_reach_count": reach.bit_count(),
        "target_scc_size": cache["target_scc_size"][target],
        "target_scc_in_degree": cache["target_scc_in_degree"][target],
        "target_scc_out_degree": cache["target_scc_out_degree"][target],
    }
    regions = {"gain_C": gain_C, "loss_C": loss_C, "change_C": change_C,
               "gain_C_cad": gain_cad, "loss_C_cad": loss_cad,
               "change_C_cad": gain_cad | loss_cad}
    fields.update({name + "_size": mask.bit_count() for name, mask in regions.items()})
    return fields, regions, reach


def make_update_id(graph_id, source, target):
    """ID of a base AF and an added attack."""
    key = "".join(f"{len(str(x))}:{x}" for x in (graph_id, source, target))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{graph_id}__u_{digest}"
