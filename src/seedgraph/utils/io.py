"""I/O utilities for JSONL and timestamps."""
from pathlib import Path
from typing import Iterator, Dict, Any
from datetime import datetime
import orjson


def write_jsonl(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Append a single JSON object to a JSONL file.

    Args:
        file_path: Path to JSONL file
        data: Dictionary to serialize
    """
    with open(file_path, "ab") as f:
        f.write(orjson.dumps(data))
        f.write(b"\n")


def read_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Read JSONL file line by line.

    Args:
        file_path: Path to JSONL file

    Yields:
        Parsed JSON objects
    """
    with open(file_path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def timestamp() -> str:
    """
    Generate ISO-formatted timestamp.

    Returns:
        ISO 8601 timestamp string
    """
    return datetime.utcnow().isoformat() + "Z"


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
