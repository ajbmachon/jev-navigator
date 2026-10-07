"""Memoize the existing pure masker for repeated offline bodies."""

from functools import lru_cache

from jev_navigator.judgments.secrets import SecretMasker


class OfflineMasker:
    def __init__(self):
        builtin = SecretMasker()
        self.mask = lru_cache(maxsize=65536)(builtin.mask)
        self.masked_values = lru_cache(maxsize=65536)(builtin.masked_values)
