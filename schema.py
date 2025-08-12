# schema.py
from pydantic import BaseModel
from typing import List

class QoSInputFlat(BaseModel):
    source_ip: str
    destination_ip: str
    protocol: str
    packet_size: int
    inter_arrival_time_ms: float
    jitter_ms: float

class QoSSequenceInput(BaseModel):
    sequence: List[QoSInputFlat]  # list of 5 items (sliding window)
