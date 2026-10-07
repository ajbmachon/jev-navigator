"""Match many exact terms in one pass, including overlapping terms and spellings."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator


class TextMatcher:
    """A literal trie with suffix links. Work follows text length plus the emitted matches."""

    def __init__(self, terms: Iterable[str]) -> None:
        self.edges: list[dict[str, int]] = [{}]
        self.suffix = [0]
        self.outputs: list[list[str]] = [[]]
        for term in dict.fromkeys(terms):
            if not term:
                continue
            node = 0
            for char in term:
                if char not in self.edges[node]:
                    self.edges[node][char] = len(self.edges)
                    self.edges.append({})
                    self.suffix.append(0)
                    self.outputs.append([])
                node = self.edges[node][char]
            self.outputs[node].append(term)
        queue = deque(self.edges[0].values())
        while queue:
            parent = queue.popleft()
            for char, node in self.edges[parent].items():
                queue.append(node)
                fallback = self.suffix[parent]
                while fallback and char not in self.edges[fallback]:
                    fallback = self.suffix[fallback]
                self.suffix[node] = self.edges[fallback].get(char, 0)
                self.outputs[node].extend(self.outputs[self.suffix[node]])

    def matches(self, text: str) -> Iterator[tuple[str, int]]:
        """Each term and its first character position, in text order. Overlaps remain distinct."""
        node = 0
        for position, char in enumerate(text):
            while node and char not in self.edges[node]:
                node = self.suffix[node]
            node = self.edges[node].get(char, 0)
            for term in self.outputs[node]:
                yield term, position - len(term) + 1
