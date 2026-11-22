"""Build the grounded prompt that constrains the LLM to extracted facts."""

NO_EVIDENCE_TEXT = "No explanation edges were available."


def format_evidence(evidence):
    """Render an evidence table as a numbered list of facts."""
    lines = [
        f"{idx + 1}. {row['generated_fact']} "
        f"(direction: {row['direction']}; importance: {row['importance_score']:.3f})"
        for idx, row in evidence.reset_index(drop=True).iterrows()
    ]

    return "\n".join(lines) if lines else NO_EVIDENCE_TEXT


def build_grounded_prompt(prediction, evidence, user_question):
    """Create an LLM prompt grounded only in prediction metadata and evidence."""
    evidence_text = format_evidence(evidence)

    return f"""You are a grounded conversational XAI assistant.

Task:
Answer the user's question about one node-classification prediction.

Prediction metadata:
- Explained node: {prediction["node_label"]} ({prediction["node_uri"]})
- Predicted class: {prediction["predicted_label"]}
- True class: {prediction["true_label"]}
- Prediction correct: {prediction["correct"]}

Grounding evidence from GNNExplainer:
{evidence_text}

User question:
{user_question}

Rules:
- Use only the evidence listed above.
- Do not invent graph facts, names, labels, or causes.
- Treat the GNNExplainer edges as the explanation evidence.
- The language model only verbalizes the evidence; it is not the explainer.
- If the evidence is weak, indirect, or hard to interpret, say so clearly.
- Keep the answer concise and understandable.
"""
