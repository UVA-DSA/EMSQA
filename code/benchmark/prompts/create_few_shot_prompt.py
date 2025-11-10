import random
import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any


def stratified_k_shot(
    items: List[Dict[str, Any]],
    k: int = 32,
    seed: int = 42,
    levels: List[str] = ["emr", "emt", "aemt", "paramedic"]
) -> List[Dict[str, Any]]:
    """
    Return k items by enforcing k//len(levels) per level, stratified by category within each level.
    Skips any items with level "NA", missing level, or category 'others'.
    """
    random.seed(seed)
    # Filter out items with disallowed categories or levels
    valid = [d for d in items
             if d.get("level") and d["level"][0] in levels
             and d.get("category", [None])[0].lower() != "others"]
    per_level = k // len(levels)
    sampled: List[Dict[str, Any]] = []

    for lvl in levels:
        pool = [d for d in valid if d["level"][0] == lvl]
        strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for d in pool:
            cat = d.get("category", ["NA"])[0]
            strata[cat].append(d)
        cats = list(strata.keys())
        if not cats:
            continue
        if len(cats) >= per_level:
            chosen = random.sample(cats, per_level)
            for cat in chosen:
                sampled.append(random.choice(strata[cat]))
        else:
            for cat in cats:
                sampled.append(random.choice(strata[cat]))
            needed = per_level - len(cats)
            leftovers = [d for grp in strata.values() for d in grp if d not in sampled]
            if leftovers and needed > 0:
                sampled.extend(random.sample(leftovers, min(needed, len(leftovers))))

    # Adjust if too few or too many
    if len(sampled) < k:
        remaining = [d for d in valid if d not in sampled]
        sampled.extend(random.sample(remaining, k - len(sampled)))
    elif len(sampled) > k:
        sampled = random.sample(sampled, k)
    return sampled


def format_blocks(
    examples: List[Dict[str, Any]]
) -> str:
    """
    Convert a list of examples into question/choices/answer blocks.
    """
    blocks = []
    for ex in examples:
        lines = [f"Question: {ex['question']}"]
        lines += ex.get("choices", [])
        lines.append(f'Answer: ["{ex.get("answer", "")}\"]')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_existing_shots(
    path: str,
    all_cases: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Read an existing .txt prompt file, extract its questions, and return matching items from all_cases,
    excluding category 'others'.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    questions = re.findall(r"^Question: (.*)$", raw, flags=re.MULTILINE)
    return [d for d in all_cases
            if d.get("question") in questions
            and d.get("category", [None])[0].lower() != "others"]


if __name__ == "__main__":
    train_path = "/scratch/zar8jw/EMS-MCQA/data/final/train_open.json"
    existing_path = "/scratch/zar8jw/EMS-MCQA/code/benchmark/prompts/16-shot.txt"
    total_k = 32

    # Load dataset
    with open(train_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    # Reconstruct existing examples (exclude 'others')
    existing_shots = parse_existing_shots(existing_path, all_cases)

    # Filter remaining (exclude existing and 'others')
    remaining = [d for d in all_cases
                 if d not in existing_shots
                 and d.get("category", [None])[0].lower() != "others"]

    # Sample additional to reach total_k
    to_sample = total_k - len(existing_shots)
    additional = stratified_k_shot(remaining, k=to_sample, seed=3407) if to_sample > 0 else []

    # Combine
    shot_examples = existing_shots + additional

    # Show statistics
    lvl_counts = Counter(ex['level'][0] for ex in shot_examples)
    cat_counts = Counter(ex['category'][0] for ex in shot_examples)
    print("Level distribution:")
    for lvl, cnt in sorted(lvl_counts.items()):
        print(f"  {lvl}: {cnt}")
    print("\nCategory distribution:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")

    # Read and clean existing prompt text
    with open(existing_path, "r", encoding="utf-8") as f:
        existing_text = f.read().rstrip()
    existing_text = re.sub(r"Question: \{question\}", "", existing_text).rstrip()

    # Format new blocks
    new_text = format_blocks(additional)

    # Merge preserving original (without placeholder)
    combined = existing_text + ("\n\n" + new_text if new_text else "") + "\n\nQuestion: {question}"

    # Save
    out_path = f"{total_k}-shot.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"\n✅ Prompt saved to '{out_path}'")
