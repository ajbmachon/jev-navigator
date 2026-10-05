"""What a run file keeps of an error message by default, computed here from the text itself."""

from __future__ import annotations

import hashlib


def digested(message: str) -> dict:
    return {"message_length": len(message), "message_sha256": hashlib.sha256(message.encode()).hexdigest()}
