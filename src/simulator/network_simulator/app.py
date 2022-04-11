from typing import List, Tuple

class Application:
    """Dummy application layer implementation."""
    def __init__(self):
        self.pkt_cnt = 0


    def has_data(self, ts: float)-> bool:
        return not (4 < ts < 5 or 10 < ts < 13)

    def get_packet(self)-> Tuple[int, int]:
        """Get packet data from application layer. Called by CC."""
        pkt_cnt = self.pkt_cnt
        self.pkt_cnt += 1
        # print(pkt_cnt, self.pkt_cnt)
        return pkt_cnt, 1500

    def feedback(self, ts: float, packet_info: List[Tuple[int, bool, float]]):
        """Return packet info to application layer. Called by CC."""
        # print(ts, packet_info)
        return
