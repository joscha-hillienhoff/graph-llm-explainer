import pytest
from rdflib import Graph, Literal, Namespace, URIRef

from graph_llm_explainer import SWRC, GraphDecoder

EX = Namespace("http://example.org/")
ONT = Namespace(SWRC)


@pytest.fixture
def tiny_graph():
    """A synthetic graph with every relation the decoder needs to handle."""
    g = Graph()
    g.bind("", ONT)

    for person, name in (("p1", "Person One"), ("p2", "Person Two")):
        g.add((EX[person], ONT.name, Literal(name)))

    for pub, title in (("pub1", "Paper A"), ("pub2", "Paper B")):
        g.add((EX[pub], ONT.title, Literal(title)))

    g.add((EX.p1, ONT.publication, EX.pub1))
    g.add((EX.p1, ONT.publication, EX.pub2))
    g.add((EX.p2, ONT.publication, EX.pub2))
    g.add((EX.pub1, ONT.author, EX.p1))
    g.add((EX.pub2, ONT.author, EX.p2))

    # Relations that are removed before training; the most frequent one
    # would shift every later relation id if it were kept in the vocabulary.
    for person in ("p1", "p2"):
        g.add((EX.group1, ONT.member, EX[person]))
        g.add((EX.group2, ONT.member, EX[person]))
        g.add((EX[person], ONT.affiliation, EX.group1))
    g.add((EX.group1, ONT.employs, EX.p1))

    return g


@pytest.fixture
def decoder(tiny_graph):
    return GraphDecoder.from_graph(tiny_graph)


@pytest.fixture
def node_ids(decoder):
    return {uri: idx for idx, uri in decoder.inv_nodes_dict.items()}


@pytest.fixture
def prediction():
    return {
        "node_label": "Person One",
        "node_uri": str(URIRef("http://example.org/p1")),
        "predicted_label": "Group One",
        "true_label": "Group One",
        "correct": True,
    }
