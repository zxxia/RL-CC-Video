import numpy as np
import pandas as pd
from typing import List, Tuple
"""
This module contains the basic components of the application, including packets and different frames
"""

G_PACKET_SIZE = 1000

class PacketInfo:
    def __init__(self, id: int, size: int):
        """
        id: the packet id
        size: the size in bytes
        """
        self._id = id
        self._size = size

    @property
    def id(self):
        return self._id

    @property
    def size(self):
        return self._size

class PacketBuffer:
    """
    a buffer maintaining all packets with ID and size
    """
    def __init__(self):
        self.buffer = []

    def add_packet(self, pkt: PacketInfo):
        """
        id: the id of the packet
        size: the size of the packet
        """
        self.buffer.append(pkt)

    def queue_len(self) -> bool:
        return len(self.buffer)

    def has_data(self) -> bool:
        return len(self.buffer) > 0

    def peek_packet(self):
        """
        Get the first packet from the packet buffer
        Returns:
            pkt: the first packet in the packet buffer
                 returns None if no packet is present
        """
        if not self.has_data():
            return None
        return self.buffer[0]

    def get_packet(self) -> PacketInfo:
        """
        Get and remove the first packet from the packet buffer
        Returns:
            pkt: the first packet in the packet buffer
                 returns None if no packet is present
        """
        if not self.has_data():
            return None
        pkt = self.buffer[0]
        self.buffer = self.buffer[1:]
        return pkt

class FrameInfo:
    """
    Maintains the send/recv information for a frame
    Provides packetizing interface and ID translation
    """
    def __init__(self, frame_id, size, psnr):
        """
        frame_id: the id of the frame
        size: the size of the frame in bytes
        """
        global G_PACKET_SIZE
        assert size < 500 * G_PACKET_SIZE, "Frame size cannot be larger than 500 G_PACKET_SIZE"
        self.frame_id = frame_id
        self.size = size
        self.psnr = psnr

        ''' when is the frame sent out and received '''
        self.sent_time = None
        self.recv_time = None

        ''' fragments who are sent but not received '''
        self.unreceived_frags = []

    def _gen_packet_id(self, frag_id):
        return self.frame_id * 500 + frag_id

    @staticmethod
    def TranslatePacketId(pkt_id):
        """
        returns frame id and pkt id
        """
        return pkt_id // 500, pkt_id % 500

    def on_sent(self, ts: float):
        """
        Called when the frame is put into the sending buffer
        Input: ts: the timestamp of the sent frame
        """
        if self.sent_time is not None:
            raise RuntimeError(f"Frame {self.frame_id} is sent twice!")
        self.sent_time = ts

    def packetize(self) -> List[PacketInfo]:
        """
        Packetize the frame into packets, can only be called once!
        Will set the packets into unreceived
        """
        global G_PACKET_SIZE
        remaining_size = self.size
        ret = []
        frag_id = 0
        while remaining_size > 0:
            pkt_size = min(remaining_size, G_PACKET_SIZE)
            pkt_id = self._gen_packet_id(frag_id)
            pkt = PacketInfo(pkt_id, pkt_size)
            ret.append(pkt)

            remaining_size -= pkt_size
            self.unreceived_frags.append(frag_id)
            frag_id += 1

        return ret

    def on_recv(self, ts: float, pkt: PacketInfo) -> bool:
        """
        Called when a packet of this frame is received
        ts: the timestamp when packet is received
        Returns:
            True if all frags are received, otherwise false
        """
        fid, frag_id = self.TranslatePacketId(pkt.id)
        if fid != self.frame_id:
            print("\033[33m Warning: this packet does not belong to the current frame\033[0m")
            return
        self.unreceived_frags.remove(frag_id)

        ''' update recv_time if the frame is just all received '''
        if self.recv_time is None and self.all_received():
            self.recv_time = ts
        return self.all_received()

    def is_sent(self) -> bool:
        """
        Returns:
            True if the frame is sent
        """
        return self.sent_time is not None

    def all_received(self) -> bool:
        """
        Returns:
            True only if the frame is already sent out and all received
        """
        return self.is_sent() and len(self.unreceived_frags) == 0

    def get_delay(self) -> float:
        """
        Returns the delay if the frame is sent and received, otherwise None
        """
        if self.sent_time is not None and self.recv_time is not None:
            return self.recv_time - self.sent_time
        return None

    def get_psnr(self) -> float:
        """
        Returns the PSNR of the frame
        """
        if self.sent_time is not None and self.recv_time is not None:
            return self.psnr
        return None

class AEFrameInfo(FrameInfo):
    def __init__(self, frame_id, size, psnr, ae_encoder):
        super(AEFrameInfo, self).__init__(frame_id, size, psnr)
        self.encoder = ae_encoder
        self.last_frag_recv_time = None

    def get_psnr(self):
        """
        Returns:
            the estimated psnr based on the received packet information
        """
        ''' compute loss '''
        tot_frags = self.size / G_PACKET_SIZE
        lost_frags = len(self.unreceived_frags)
        loss_rate = lost_frags / tot_frags

        ''' query AE encoder using loss and frame id '''
        psnr = self.encoder.query_frame_with_size(self.frame_id, self.size, loss_rate)
        return psnr
    
    def on_recv(self, ts: float, pkt: PacketInfo) -> bool:
        """
        Called when a packet of this frame is received
        ts: the timestamp when packet is received
        Returns:
            True if all frags are received, otherwise false
        """
        ret = FrameInfo.on_recv(self, ts, pkt)
        if not self.all_received():
            self.last_frag_recv_time = ts
            print("\033[31m [{}] pkt {} received, updated frag_recv_time={}, now delay is {}\033[0m".format(self.frame_id, pkt.id, self.last_frag_recv_time, self.get_delay()))
        return ret

    def get_delay(self) -> float:
        if self.sent_time is not None and self.recv_time is not None:
            return self.recv_time - self.sent_time
        elif self.sent_time is not None and self.last_frag_recv_time is not None:
            return self.last_frag_recv_time - self.sent_time
        return None
