from pathlib import Path
import re
import networkx as nx

#parse graphs from tgf or apx
APX_ATOM_RE = re.compile(
    r"^\s*(?P<predicate>arg|att)\s*\((?P<args>[^)]*)\)\s*\.\s*$"
)

#tgf to nx
def tgf_to_nx_graph(file_path):
    file_path = Path(file_path)
    graph = nx.DiGraph()
    edge_section = False
    seen_separator = False

    with file_path.open("r", encoding="utf-8", errors="strict") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line == "#":
                if seen_separator:
                    raise ValueError(
                        f"Multiple '#' separators in TGF file at {file_path}:{line_number}"
                    )
                edge_section = True
                seen_separator = True
                continue

            parts = line.split()

            if not edge_section:
                if len(parts) != 1:
                    raise ValueError(
                        f"Invalid TGF node line at {file_path}:{line_number}: {line}"
                    )
                graph.add_node(parts[0])
            else:
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid TGF edge line at {file_path}:{line_number}: {line}"
                    )
                graph.add_edge(parts[0], parts[1])

    if not seen_separator:
        raise ValueError(f"TGF file has no '#' separator: {file_path}")

    return graph

#apx to nx
def apx_to_nx_graph(file_path):
    file_path = Path(file_path)
    graph = nx.DiGraph()

    with file_path.open("r", encoding="utf-8", errors="strict") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            match = APX_ATOM_RE.match(line)
            if not match:
                raise ValueError(
                    f"Invalid APX line at {file_path}:{line_number}: {line}"
                )

            predicate = match.group("predicate")
            args = match.group("args")

            if predicate == "arg":
                arg_name = args.strip()
                if not arg_name:
                    raise ValueError(
                        f"Empty argument name at {file_path}:{line_number}: {line}"
                    )
                graph.add_node(arg_name)

            elif predicate == "att":
                parts = [x.strip() for x in args.split(",")]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(
                        f"Invalid APX attack atom at {file_path}:{line_number}: {line}"
                    )
                graph.add_edge(parts[0], parts[1])

    return graph

#read tgf and apx AF files and return nx graph
def read_graph(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".tgf":
        return tgf_to_nx_graph(file_path)

    if suffix == ".apx":
        return apx_to_nx_graph(file_path)

    raise ValueError(f"Unsupported graph format: {suffix}")