"""I/O utilities for JSONL and timestamps."""
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime
import orjson
import gzip
import zstandard as zstd


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


def write_jsonl_compressed(file_path: Path, data: Dict[str, Any], compression: str = "zstd") -> None:
    """
    Append a single JSON object to a compressed JSONL file.

    Args:
        file_path: Path to compressed JSONL file
        data: Dictionary to serialize
        compression: Compression type ("zstd" or "gzip")
    """
    json_bytes = orjson.dumps(data) + b"\n"

    if compression == "zstd":
        mode = "ab"
        if not file_path.exists():
            # Create new compressed file
            with open(file_path, "wb") as f:
                cctx = zstd.ZstdCompressor()
                f.write(cctx.compress(json_bytes))
        else:
            # Append to existing (decompress, append, recompress)
            # For streaming, we'll use a simpler approach: append compressed chunks
            with open(file_path, "ab") as f:
                cctx = zstd.ZstdCompressor()
                f.write(cctx.compress(json_bytes))
    elif compression == "gzip":
        with gzip.open(file_path, "ab") as f:
            f.write(json_bytes)
    else:
        raise ValueError(f"Unsupported compression: {compression}")


def read_jsonl_compressed(file_path: Path, compression: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """
    Read compressed JSONL file line by line.

    Args:
        file_path: Path to compressed JSONL file
        compression: Compression type ("zstd", "gzip", or None for auto-detect)

    Yields:
        Parsed JSON objects
    """
    # Auto-detect compression from extension
    if compression is None:
        if str(file_path).endswith(".zst") or str(file_path).endswith(".zstd"):
            compression = "zstd"
        elif str(file_path).endswith(".gz"):
            compression = "gzip"
        else:
            compression = None

    if compression == "zstd":
        with open(file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read().decode("utf-8")
                for line in text_stream.splitlines():
                    line = line.strip()
                    if line:
                        yield orjson.loads(line)
    elif compression == "gzip":
        with gzip.open(file_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield orjson.loads(line)
    else:
        # Uncompressed
        yield from read_jsonl(file_path)
