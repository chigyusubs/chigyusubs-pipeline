"""Glossary loading utilities."""


def load_glossary_names(glossary_path: str) -> list[str]:
    """Extract names/terms from glossary TSV for prompts."""
    entries = []
    with open(glossary_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] != "source":
                entries.append(f"{parts[0]} ({parts[1]})")
    return entries
