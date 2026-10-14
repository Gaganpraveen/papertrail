"""Bounded local PDF ingestion. Original filenames are display labels, never paths."""

import hashlib
import re
from pathlib import Path

from papertrail.errors import ParseError
from papertrail.schema import Paper


def display_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    if len(name) > 240 and name.lower().endswith(".pdf"):
        return name[:236] + name[-4:]
    return name[:240] or "Uploaded document.pdf"


def copy_upload(
    source: Path, destination: Path, filename: str, max_bytes: int
) -> tuple[Paper, str]:
    """Copy bytes to a fixed internal path and derive a content identity."""
    name = display_filename(filename)
    if not name.lower().endswith(".pdf"):
        raise ParseError("Choose a PDF file with a .pdf filename. Other formats are not supported.")
    digest = hashlib.sha256()
    copied = 0
    temporary = destination.with_suffix(".pdf.tmp")
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            if incoming.read(5) != b"%PDF-":
                raise ParseError("The uploaded file is not a PDF. Choose a text-readable PDF file.")
            incoming.seek(0)
            while block := incoming.read(64 * 1024):
                copied += len(block)
                if copied > max_bytes:
                    raise ParseError(
                        f"The uploaded PDF exceeds the {max_bytes // (1024 * 1024)} MiB limit. Choose a smaller document."
                    )
                digest.update(block)
                outgoing.write(block)
        temporary.replace(destination)
    except OSError as exc:
        raise ParseError("The uploaded PDF could not be saved. Choose the file again.") from exc
    finally:
        temporary.unlink(missing_ok=True)
    sha256 = digest.hexdigest()
    paper = Paper(
        title=name,
        source="upload",
        document_id=f"upload:{sha256}",
        source_filename=name,
    )
    return paper, sha256
