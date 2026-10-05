"""Made-up values with the shapes of secrets the cut-before-mask audit probed; none was ever a credential."""

from __future__ import annotations

import json
from collections.abc import Mapping

from jev_navigator.judgments.secrets import SecretMasker, mask_request

SECRET_SHAPES = {
    "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    + "Qm9yZWFsUGluZVRlc3RWYWx1ZU9ubHlOb3RBQ3JlZGVudGlhbEp1c3RBUHJvYmUx"
    + ".Zk8qLm3NpR7sTv2WxY5zAb9CdE1fGh4JkL6mNo",
    "password-with-symbols": "Kq8mLx2PzR7v!Wn4TsB9cHd3@FgJ6aE1yUo5#Rt0Xb3Nc8Vm2",
    "dotted-key": "Lp4Kx9Qm2Rz7Vn3Ts8Wb5Yc1.Hd6Fg0Jk4Ma9Nb2Pc7Qd3R.e8Sf1Tg5Uh0Vi6Wj2Xk9Yl4Z",
    "hex": "9f3a1c7e5b2d8046af1e3c9b7d5a2f80" + "c4e6a8b0d2f41357968ace0bdf135792",
    "alphanumeric": "Vq7Lm2Xz9Rk4Tn8Wb3Yc6Hd1Fg5Jp0Ns2Qa7Ue",
}
COPY_VALUE = "Wb3Yc6Hd1Fg5Jp0Ns2Qa7UeVq7Lm2Xz9"
"""A 32-character key, as the audit's M2 probes passed it bare in code whose key line was cut away."""


def pieces(value: str, size: int = 6) -> set[str]:
    return {value[at : at + size] for at in range(len(value) - size + 1)}


def sent_pieces(value: str, request: Mapping) -> set[str]:
    """The pieces of ``value`` still in ``request`` after the judge's own masking."""
    sent = json.dumps(mask_request(request, {}, SecretMasker())[0])
    return {piece for piece in pieces(value) if piece in sent}
