from graph_llm_explainer import build_grounded_prompt
from graph_llm_explainer.prompts import NO_EVIDENCE_TEXT


def evidence_lines(prompt):
    block = prompt.split("Grounding evidence from GNNExplainer:\n")[1]
    return block.split("\n\nUser question:")[0].splitlines()


def test_prompt_contains_only_the_extracted_facts(decoder, node_ids, prediction):
    p1 = node_ids["http://example.org/p1"]
    pub1 = node_ids["http://example.org/pub1"]
    table = decoder.build_evidence_table(
        {
            "edges": [
                {"source_id": p1, "target_id": pub1, "edge_type": 0, "importance": 0.5}
            ]
        }
    )

    prompt = build_grounded_prompt(prediction, table, "Why this class?")

    assert evidence_lines(prompt) == [
        "1. Person One --:publication--> Paper A "
        "(direction: forward; importance: 0.500)"
    ]
    # Graph content outside the evidence must not leak into the prompt
    assert "Paper B" not in prompt
    assert "Person Two" not in prompt


def test_prompt_states_missing_evidence(decoder, prediction):
    table = decoder.build_evidence_table({"edges": []})

    prompt = build_grounded_prompt(prediction, table, "Why this class?")

    assert table.empty
    assert evidence_lines(prompt) == [NO_EVIDENCE_TEXT]
