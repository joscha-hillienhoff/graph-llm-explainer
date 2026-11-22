from graph_llm_explainer import (
    SWRC,
    build_model_triples,
    build_relation_vocabulary,
)


def test_vocabulary_excludes_target_and_shortcut_relations(tiny_graph):
    relations = build_relation_vocabulary(build_model_triples(tiny_graph))

    # Ordered by descending frequency, ties broken by URI
    assert [str(r).removeprefix(SWRC) for r in relations] == [
        "publication",
        "author",
        "name",
        "title",
    ]


def test_edge_to_fact_restores_rdf_direction(decoder, node_ids):
    pub1 = node_ids["http://example.org/pub1"]
    p1 = node_ids["http://example.org/p1"]
    author = 1  # relation id of ":author" in the vocabulary above

    forward = decoder.edge_to_fact(
        {
            "source_id": pub1,
            "target_id": p1,
            "edge_type": 2 * author,
            "importance": 0.8,
        }
    )
    inverse = decoder.edge_to_fact(
        {
            "source_id": p1,
            "target_id": pub1,
            "edge_type": 2 * author + 1,
            "importance": 0.4,
        }
    )

    assert forward["generated_fact"] == "Paper A --:author--> Person One"
    assert forward["direction"] == "forward"
    assert inverse["generated_fact"] == "Paper A --:author--> Person One"
    assert inverse["direction"] == "inverse"


def test_decoder_uses_the_relation_vocabulary_of_the_trained_model(decoder):
    # With only `affiliation` excluded, `member` would take index 0 and every
    # later relation would decode to the wrong predicate.
    predicates = [
        decoder.compact_term(decoder.decode_edge_type(2 * relation_id)[0])
        for relation_id in range(4)
    ]

    assert predicates == [":publication", ":author", ":name", ":title"]


def test_evidence_table_ranks_by_importance_and_keeps_top_k(decoder, node_ids):
    p1 = node_ids["http://example.org/p1"]
    edges = [
        {
            "source_id": p1,
            "target_id": node_ids["http://example.org/pub1"],
            "edge_type": 0,
            "importance": 0.2,
        },
        {
            "source_id": p1,
            "target_id": node_ids["http://example.org/pub2"],
            "edge_type": 0,
            "importance": 0.9,
        },
    ]

    table = decoder.build_evidence_table({"edges": edges}, top_k=1)

    assert list(table["generated_fact"]) == ["Person One --:publication--> Paper B"]
