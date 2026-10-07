"""Diagnostic positions in prepared requests, accounting for split units."""

import math

from jev_navigator.index.units import items_to_judge


def request_rank(units, file, line):
    position = 0
    for unit in units:
        for item in items_to_judge(unit):
            position += 1
            if item.file == file and any(start <= line <= end for start, end in item.ranges):
                return math.ceil(position / 16)
    return None
