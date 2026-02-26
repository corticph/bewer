from enum import IntEnum

__all__ = ["OpType"]


class OpType(IntEnum):
    MATCH = 0
    INSERT = 1
    DELETE = 2
    SUBSTITUTE = 3
