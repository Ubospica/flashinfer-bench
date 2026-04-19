"""Helpers for building and parsing multipart responses."""

from __future__ import annotations

import json
import re
import uuid
from typing import Dict, Tuple


def _encode_part(boundary_bytes: bytes, name: str, content_type: str, payload: bytes) -> bytes:
    """Encode a single multipart part.

    Parameters
    ----------
    boundary_bytes
        Multipart boundary encoded as ASCII bytes.
    name
        Multipart part name.
    content_type
        MIME type for the part payload.
    payload
        Raw part payload bytes.

    Returns
    -------
    bytes
        Encoded multipart fragment including headers and trailing CRLF.
    """
    return (
        b"--"
        + boundary_bytes
        + b"\r\n"
        + f'Content-Disposition: form-data; name="{name}"\r\n'.encode("utf-8")
        + f"Content-Type: {content_type}\r\n\r\n".encode("ascii")
        + payload
        + b"\r\n"
    )


def build_success_response(
    result_payload: dict, binary_parts: Dict[str, bytes]
) -> Tuple[bytes, str]:
    """Build a multipart response body and content type.

    Parameters
    ----------
    result_payload
        JSON-serializable payload stored in the `result` multipart part.
    binary_parts
        Named binary multipart parts keyed by part name.

    Returns
    -------
    tuple[bytes, str]
        Encoded multipart body and corresponding HTTP `Content-Type` header value.
    """
    boundary = f"flashinfer-bench-{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("ascii")
    chunks: list[bytes] = []
    chunks.append(
        _encode_part(
            boundary_bytes, "result", "application/json", json.dumps(result_payload).encode("utf-8")
        )
    )
    for part_name, payload in binary_parts.items():
        chunks.append(_encode_part(boundary_bytes, part_name, "application/octet-stream", payload))
    chunks.append(b"--" + boundary_bytes + b"--\r\n")
    body = b"".join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


_BOUNDARY_RE = re.compile(r"boundary=([^\s;]+)")
_NAME_RE = re.compile(r'name="([^"]+)"')


def parse_multipart_response(content_type: str, body: bytes) -> Tuple[dict, Dict[str, bytes]]:
    """Parse a multipart response produced by `build_success_response`.

    Parameters
    ----------
    content_type
        HTTP Content-Type header value containing the boundary.
    body
        Raw multipart response body bytes.

    Returns
    -------
    tuple[dict, dict[str, bytes]]
        Parsed JSON result payload and named binary parts.
    """
    boundary_match = _BOUNDARY_RE.search(content_type)
    if boundary_match is None:
        raise ValueError(f"No boundary in Content-Type: {content_type}")
    delimiter = b"--" + boundary_match.group(1).encode("ascii")

    result: dict | None = None
    binary_parts: Dict[str, bytes] = {}

    for segment in body.split(delimiter):
        if not segment or segment in (b"\r\n", b"--\r\n", b"--"):
            continue
        header_end = segment.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = segment[:header_end].decode("utf-8", errors="replace")
        payload = segment[header_end + 4 :]
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]

        name_match = _NAME_RE.search(header_block)
        if name_match is None:
            continue
        name = name_match.group(1)

        if name == "result":
            result = json.loads(payload)
        else:
            binary_parts[name] = payload

    if result is None:
        raise ValueError("Missing 'result' part in multipart response")
    return result, binary_parts
