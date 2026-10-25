def binary_question(instructions: str, *, question_id: str = "decision",
                    evidence: str = "ECG evidence: <image>", swap: bool = False) -> dict:
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("nonempty instructions required")
    if not isinstance(evidence, str) or evidence.count("<image>") != 1:
        raise ValueError("evidence must contain exactly one <image> marker")
    order = ["false", "true"] if swap else ["true", "false"]
    return {
        "id": question_id,
        "type": "noul",
        "state": evidence,
        "instructions": instructions,
        "options": [{"id": key} for key in order],
        "semantic_option_order": order,
    }
