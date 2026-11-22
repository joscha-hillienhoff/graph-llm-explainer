"""Decode GNNExplainer edges into readable RDF facts.

The R-GCN encodes every relation ``r`` as two edge types: ``2 * r`` for the
original direction and ``2 * r + 1`` for the inverse. Decoding a saved
explanation therefore needs exactly the relation vocabulary that was used to
train the model. The helpers in this module are the single place where that
vocabulary is built, so training and decoding cannot drift apart.
"""

import re
from collections import Counter

import pandas as pd
from rdflib import BNode, Literal, URIRef

SWRC = "http://swrc.ontoware.org/ontology#"

DEFAULT_NAME_PREDICATES = (
    URIRef(SWRC + "name"),
    URIRef(SWRC + "title"),
    URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
)


def model_excluded_predicates(exclude_member=True, exclude_employs=True):
    """Return the predicates removed from the graph before R-GCN training.

    ``affiliation`` is the prediction target. ``member`` and ``employs`` are
    removed as well because they partially encode the same information.
    """
    excluded = {URIRef(SWRC + "affiliation")}

    if exclude_member:
        excluded.add(URIRef(SWRC + "member"))

    if exclude_employs:
        excluded.add(URIRef(SWRC + "employs"))

    return excluded


def build_model_triples(graph, excluded=None):
    """Return the triples of ``graph`` that are visible to the R-GCN."""
    if excluded is None:
        excluded = model_excluded_predicates()

    return [
        (subject, predicate, obj)
        for subject, predicate, obj in graph
        if predicate not in excluded
    ]


def build_relation_vocabulary(model_triples):
    """Return relations ordered by descending frequency, then by URI."""
    frequency = Counter(predicate for _, predicate, _ in model_triples)

    return sorted(frequency, key=lambda relation: (-frequency[relation], str(relation)))


class GraphDecoder:
    """Turn numeric node and edge-type ids into readable RDF facts."""

    def __init__(
        self,
        graph,
        inv_nodes_dict,
        relations,
        name_predicates=DEFAULT_NAME_PREDICATES,
    ):
        """Store the graph and the id mappings used by the trained model.

        ``inv_nodes_dict`` maps a node id to its URI string and ``relations``
        is the list returned by :func:`build_relation_vocabulary`.
        """
        self.graph = graph
        self.inv_nodes_dict = inv_nodes_dict
        self.inv_relations_dict = dict(enumerate(relations))
        self.name_predicates = tuple(name_predicates)

    @classmethod
    def from_graph(cls, graph, excluded=None, name_predicates=DEFAULT_NAME_PREDICATES):
        """Build a decoder that matches the model trained on ``graph``."""
        nodes = sorted(set(graph.subjects()).union(graph.objects()), key=str)
        inv_nodes_dict = dict(enumerate(str(node) for node in nodes))
        relations = build_relation_vocabulary(build_model_triples(graph, excluded))

        return cls(graph, inv_nodes_dict, relations, name_predicates)

    def compact_term(self, term):
        """Convert an RDF term into a compact, human-readable string."""
        if term is None:
            return ""

        if isinstance(term, Literal):
            return str(term)

        if isinstance(term, BNode):
            return f"_:{term}"

        value = str(term)

        if value.startswith(("http://", "https://")):
            try:
                return self.graph.namespace_manager.normalizeUri(URIRef(value))
            except Exception:
                tail = re.split(r"[#/]", value.rstrip("/"))[-1]
                return tail or value

        return value

    def first_name_for_uri(self, uri):
        """Return a readable name, title, or label for an RDF URI if available."""
        uri = str(uri)

        if not uri.startswith(("http://", "https://")):
            return None

        term = URIRef(uri)

        for predicate in self.name_predicates:
            values = [str(value) for value in self.graph.objects(term, predicate)]
            if values:
                return values[0]

        return None

    def readable_node_label(self, node_id):
        """Convert a numeric node id into a readable RDF node label."""
        uri = self.inv_nodes_dict.get(int(node_id), str(node_id))
        return self.first_name_for_uri(uri) or self.compact_term(uri)

    def readable_label(self, label_uri):
        """Convert a class URI into a readable class label."""
        return self.first_name_for_uri(label_uri) or self.compact_term(label_uri)

    def decode_edge_type(self, edge_type):
        """Decode an R-GCN edge type into its RDF predicate and direction."""
        edge_type = int(edge_type)

        relation_id = edge_type // 2
        direction = "forward" if edge_type % 2 == 0 else "inverse"
        predicate = self.inv_relations_dict[relation_id]

        return predicate, direction

    def edge_to_fact(self, edge):
        """Convert one saved GNNExplainer edge into a readable RDF fact."""
        src_id = int(edge.get("source_id", edge.get("src", edge.get("source"))))
        dst_id = int(edge.get("target_id", edge.get("dst", edge.get("target"))))
        edge_type = int(
            edge.get("edge_type", edge.get("relation", edge.get("relation_id")))
        )
        importance = float(
            edge.get("importance", edge.get("score", edge.get("edge_mask", 0.0)))
        )

        predicate, direction = self.decode_edge_type(edge_type)

        source_label = self.readable_node_label(src_id)
        target_label = self.readable_node_label(dst_id)
        predicate_label = self.compact_term(predicate)

        # Restore the original RDF direction for inverse model edges
        if direction == "inverse":
            rdf_subject, rdf_object = target_label, source_label
        else:
            rdf_subject, rdf_object = source_label, target_label

        return {
            "source_id": src_id,
            "target_id": dst_id,
            "edge_type": edge_type,
            "source_label": source_label,
            "target_label": target_label,
            "predicate": predicate_label,
            "direction": direction,
            "importance_score": importance,
            "generated_fact": f"{rdf_subject} --{predicate_label}--> {rdf_object}",
        }

    def build_evidence_table(self, explanation, top_k=10):
        """Build a ranked evidence table from a saved GNNExplainer record."""
        edges = explanation.get("edges", explanation.get("explanation_edges", []))
        evidence = pd.DataFrame([self.edge_to_fact(edge) for edge in edges])

        if evidence.empty:
            return evidence

        return (
            evidence.sort_values("importance_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
