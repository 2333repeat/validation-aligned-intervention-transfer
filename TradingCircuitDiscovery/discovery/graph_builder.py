from __future__ import annotations


def build_circuit_graph(concept: str, circuit_name: str, neurons: list[str], behavior: str) -> dict:
    nodes = [{"id": concept, "type": "concept"}, {"id": circuit_name, "type": "circuit"}]
    nodes.extend({"id": neuron, "type": "neuron"} for neuron in neurons)
    nodes.append({"id": behavior, "type": "behavior"})
    edges = []
    previous = concept
    for neuron in neurons:
        edges.append({"source": previous, "target": neuron})
        previous = neuron
    edges.append({"source": previous, "target": circuit_name})
    edges.append({"source": circuit_name, "target": behavior})
    return {"nodes": nodes, "edges": edges}
