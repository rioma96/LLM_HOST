def build_relation_prompt(
    sentence: str,
    entity_1: str,
    entity_2: str,
    relation_name: str,
    relation_description: str,
) -> str:
    description = relation_description.strip() or "No extra description provided."
    return (
        "You are an information extraction oracle. "
        "Decide if a target relation is explicitly or implicitly supported by the sentence.\n\n"
        f"Sentence: {sentence}\n"
        f"Entity 1: {entity_1}\n"
        f"Entity 2: {entity_2}\n"
        f"Target relation: {relation_name}\n"
        f"Relation description: {description}\n\n"
        "Rules:\n"
        "1) Return relation_present=true only if evidence in sentence supports the relation.\n"
        "2) If evidence is weak/ambiguous, return false.\n"
        "3) Be strict and avoid hallucinating facts.\n"
        "4) Confidence must be a float in [0,1].\n\n"
        "Return ONLY valid JSON with keys: relation_present, confidence, reason_short."
    )
