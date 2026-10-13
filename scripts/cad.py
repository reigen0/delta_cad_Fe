"""Read acceptance status from acceptance files or compute DC-PR for all args using mutoksia."""
from pathlib import Path
import subprocess
from parse_graphs import read_graph


def read_args(text):
    return {x.strip() for x in text.replace("\n", ",").split(",") if x.strip()}


def join_args(values):
    return ",".join(sorted(values))


def read_cad(path):
    return read_args(Path(path).read_text(encoding="utf-8"))


def compute_cad(graph_path, solver, timeout=60, output_path=None):
    """Return (complete set, None, computed) or (None, timeout argument, False)."""
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists():
            return read_cad(output_path), None, False

    graph_path = Path(graph_path)
    graph = read_graph(graph_path)
    accepted = set()
    for argument in graph:
        try:
            result = subprocess.run(
                [str(solver), "-p", "DC-AD", "-f", str(graph_path),
                 "-fo", graph_path.suffix[1:], "-a", str(argument)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, argument, False
        if result.returncode != 0:
            raise RuntimeError(f"Solver failed: {graph_path.name}, {argument}")
        tokens = (result.stdout + " " + result.stderr).upper().split()
        if "YES" in tokens:
            accepted.add(argument)
        elif "NO" not in tokens:
            raise RuntimeError(f"Unexpected solver output: {graph_path.name}, {argument}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(join_args(accepted), encoding="utf-8")
        temporary.replace(output_path)
    return accepted, None, True
