import heapq
import random
from typing import List

from simulator.network_simulator.constants import EVENT_TYPE_ACK, EVENT_TYPE_SEND
from simulator.network_simulator.packet import Packet
from simulator.network_simulator.link import Link
from simulator.network_simulator.sender import SenderType

USE_LATENCY_NOISE = False
# USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.01

# DEBUG = True
DEBUG = False


class Network:
    """Network model"""

    def __init__(self, senders: List[SenderType], links: List[Link],
                 record_pkt_log: bool = False):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.record_pkt_log = record_pkt_log
        self.pkt_log = []
        self.extra_delays = []  # time used to put packet onto the network
        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()

            sender.schedule_send(True)

    def add_packet(self, pkt: Packet) -> None:
        """Add a packet to the packet event queue."""
        heapq.heappush(self.q, pkt)

    def reset(self) -> None:
        self.cur_time = 0.0
        self.q = []
        for link in self.links:
            link.reset()
        for sender in self.senders:
            sender.reset()
        self.queue_initial_packets()
        self.pkt_log = []
        self.extra_delays = []  # time used to put packet onto the network

    def get_cur_time(self) -> float:
        """Return current network time."""
        return self.cur_time

    def run(self, dur: float):
        """Run the network with specified duration."""
        for sender in self.senders:
            sender.reset_obs()
        end_time = min(self.cur_time + dur, self.links[0].trace.timestamps[-1])
        self.extra_delays = []  # time used to put packet onto the network
        while True:
            pkt = self.q[0]
            # pkt.debug_print()
            # use got_data here to make sure aurora receives at least a pkt ack
            # in MI at the beginning of the connection. got_data does not
            # affect other congestion controls
            # if pkt.sender.got_data and pkt.ts >= end_time and pkt.event_type == EVENT_TYPE_SEND:
            if pkt.sender.stop_run(pkt, end_time):
                end_time = pkt.ts
                self.cur_time = end_time
                break
            pkt = heapq.heappop(self.q)

            self.cur_time = pkt.ts
            push_new_event = False
            if (len(self.q) == 0 and not pkt.sender.app.has_data(self.cur_time)):
                self.add_packet(Packet(self.get_cur_time() + 0.001, pkt.sender, 0, 0))
            elif (len(self.q) == 0 and pkt.sender.app.has_data(self.cur_time)):# and self.cur_time !=0 ):
                pkt.sender.schedule_send(on_ack=True)
            if pkt.pkt_size == 0:
                if pkt.sender.app.has_data(self.cur_time):
                    pkt.sender.schedule_send(on_ack=True)
                continue
            # debug_print("Got %d event %s, to link %d, latency %f at time %f, "
            #             "next_hop %d, dropped %s, event_q length %f, "
            #             "sender rate %f, duration: %f, queue_size: %f, "
            #             "rto: %f, cwnd: %f, ssthresh: %f, sender rto %f, "
            #             "pkt in flight %d, wait time %d" % (
            #                 event_id, event_type, next_hop, cur_latency,
            #                 event_time, next_hop, dropped, len(self.q),
            #                 sender.rate, dur, self.links[0].queue_size,
            #                 rto, sender.cwnd, sender.ssthresh, sender.rto,
            #                 int(sender.bytes_in_flight/BYTES_PER_PACKET),
            #                 sender.pkt_loss_wait_time))
            sender = pkt.sender
            if pkt.event_type == EVENT_TYPE_ACK:
                if pkt.next_hop == len(self.links):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                    # sender.on_packet_lost(cur_latency)
                    # if sender.rto >= 0 and pkt.cur_latency > sender.rto and sender.pkt_loss_wait_time <= 0:
                    #     sender.timeout()
                    #     pkt.drop()
                    if pkt.dropped:
                        sender.on_packet_lost(pkt)
                        if self.record_pkt_log:
                            self.pkt_log.append([pkt.pkt_id, pkt.sent_time, -1, pkt.pkt_size])
                    else:
                        sender.on_packet_acked(pkt)
                else:  # in acklink
                    if self.record_pkt_log:
                        self.pkt_log.append([pkt.pkt_id, pkt.sent_time, pkt.cur_latency, pkt.pkt_size])
                    link_prop_latency = self.links[pkt.next_hop].get_cur_propagation_latency(
                        self.cur_time)
                    # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                    # if USE_LATENCY_NOISE:
                    # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    # TODO: add delay noise of acklink
                    pkt.add_propagation_delay(link_prop_latency, 'acklink')
                    pkt.next_hop += 1
                    push_new_event = True
            elif pkt.event_type == EVENT_TYPE_SEND:  # in datalink
                if pkt.next_hop == 0:
                    if sender.on_packet_sent(pkt):
                        push_new_event = True
                    else:
                        continue
                else:
                    push_new_event = True

                if pkt.next_hop == sender.dest:
                    pkt.event_type = EVENT_TYPE_ACK

                link_prop_latency, q_delay = self.links[pkt.next_hop].get_cur_latency(
                    self.cur_time)
                # if USE_LATENCY_NOISE:
                # link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                # link_latency += self.env.current_trace.get_delay_noise(
                #     self.cur_time, self.links[pkt.next_hop].get_bandwidth(self.cur_time)) / 1000
                # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                # rand = random.uniform(0, 1)
                pkt.add_propagation_delay(link_prop_latency, 'datalink')
                # if rand > 0.9:
                #     noise = random.uniform(0.0, self.links[pkt.next_hop].trace.delay_noise) / 1000
                #     pkt.add_propagation_delay(noise)
                pkt.add_queue_delay(q_delay, 'datalink')
                pkt.add_transmission_delay(1 / self.links[0].get_bandwidth(self.cur_time), 'datalink')
                if not self.links[pkt.next_hop].packet_enters_link(self.cur_time):
                    pkt.drop()
                self.extra_delays.append(
                    1 / self.links[pkt.next_hop].get_bandwidth(self.cur_time))
                pkt.next_hop += 1
                # if not pkt.dropped:
                #     sender.queue_delay_samples.append(new_event_queue_delay)

            if push_new_event:
                heapq.heappush(self.q, pkt)
        return
