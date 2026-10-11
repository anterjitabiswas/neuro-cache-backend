"""Trace file loader utility for SMARTCACHE AI."""
from pathlib import Path
from typing import List, Union


def load_trace_from_file(filepath: Union[str, Path]) -> List[int]:
    """Load a trace of integer IDs from a text or CSV file.

    Supports:
      - Line-separated numbers
      - Comma-separated numbers
      - Space-separated numbers
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    tokens: List[str] = []
    # If commas exist, replace with whitespace
    normalized = content.replace(",", " ").replace("\t", " ")
    for token in normalized.split():
        token = token.strip()
        if token:
            try:
                tokens.append(int(token))
            except ValueError:
                # If non-integer (e.g. hash / block hex), hash to int or skip headers
                try:
                    tokens.append(int(token, 16))
                except ValueError:
                    tokens.append(hash(token))

    return tokens
