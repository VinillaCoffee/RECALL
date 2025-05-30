from enum import Enum


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    MTR = "multi_timescale"
