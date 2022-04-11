import time
from typing import Any

from simulator.network_simulator.constants import BYTES_PER_PACKET, EVENT_TYPE_SEND
from simulator.network_simulator import sender


class Packet:
    """Packet event in simulator."""

    def __init__(self, ts: float, sender: "sender.Sender", pkt_id: int, pkt_size: int = BYTES_PER_PACKET):
        self.ts = ts
        self.sent_time = ts
        self.dropped = False
        self.sender = sender
        self.event_type = EVENT_TYPE_SEND
        self.next_hop = 0
        self.pkt_id = pkt_id
        self.queue_delay = 0.0
        self.propagation_delay = 0.0
        self.transmission_delay = 0.0
        self.pkt_size = pkt_size # bytes
        self.real_ts = time.time()
        self.app_data = {}
        self.datalink_delay = 0.0
        self.acklink_delay = 0.0

    def drop(self) -> None:
        """Mark packet as dropped."""
        self.dropped = True

    def add_transmission_delay(self, extra_delay: float, link: str) -> None:
        """Add to the transmission delay and add to the timestamp too."""
        self.transmission_delay += extra_delay
        self.ts += extra_delay
        if link == 'datalink':
            self.datalink_delay += extra_delay
        elif link == 'acklink':
            self.acklink_delay += extra_delay
        else:
            raise NotImplementedError

    def add_propagation_delay(self, extra_delay: float, link: str) -> None:
        """Add to the propagation delay and add to the timestamp too."""
        self.propagation_delay += extra_delay
        self.ts += extra_delay
        if link == 'datalink':
            self.datalink_delay += extra_delay
        elif link == 'acklink':
            self.acklink_delay += extra_delay
        else:
            raise NotImplementedError

    def add_queue_delay(self, extra_delay: float, link: str) -> None:
        """Add to the queue delay and add to the timestamp too."""
        self.queue_delay += extra_delay
        self.ts += extra_delay
        if link == 'datalink':
            self.datalink_delay += extra_delay
        elif link == 'acklink':
            self.acklink_delay += extra_delay
        else:
            raise NotImplementedError

    @property
    def cur_latency(self) -> float:
        """Return Current latency experienced.

        Latency = propagation_delay + queue_delay
        """
        return self.queue_delay + self.propagation_delay + self.transmission_delay

    @property
    def rtt(self) -> float:
        return self.cur_latency

    # override the comparison operator
    def __lt__(self, nxt):
        if self.ts == nxt.ts:
            return self.pkt_id < nxt.pkt_id
        return self.ts < nxt.ts

    def debug_print(self):
        print("Event {}: ts={}, type={}, dropped={}, {}".format(self.pkt_id, self.ts, self.event_type, self.dropped, self.app_data))

    def add_app_data(self, k: str, v: Any):
        self.app_data[k] = v

    def get_app_data(self, k: str):
        return self.app_data[k]
