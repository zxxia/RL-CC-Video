import pandas as pd
import numpy as np
from typing import List, Tuple

G_PACKET_SIZE = 1000

class Application:
    """Dummy application layer implementation."""
    def __init__(self):
        self.pkt_cnt = 0

    def queue_len(self, ts: float) -> int:
        return 1000

    def has_data(self, ts: float)-> bool:
        return not (4 < ts < 5 or 10 < ts < 13)

    def get_packet(self)-> Tuple[int, int]:
        """Get packet data from application layer. Called by CC."""
        pkt_cnt = self.pkt_cnt
        self.pkt_cnt += 1
        # print(pkt_cnt, self.pkt_cnt)
        return pkt_cnt, 1500

    def feedback(self, ts: float, packet_info: List[Tuple[int, bool, float, float, int, float]]):
        """Return packet info to application layer. Called by CC.

        Args
            ts: current timestamp in second.
            packet_info: list of (pkt id, dropped, one way delay (s),
                                  recv timestamp (s), pkt size (bytes),
                                  target bitrate (bytes/s))
        """
        # print(ts, packet_info)
        return

'''
=============================
    IMPLEMENTATIONS
=============================
'''
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

class AEFrameInfo(FrameInfo):
    def __init__(self, frame_id, size, psnr, ae_encoder):
        super(AEFrameInfo, self).__init__(frame_id, size, psnr)
        self.encoder = ae_encoder

    def estimate_psnr(self):
        """
        Returns:
            the estimated psnr based on the received packet information
        """
        # TODO: here!
        pass

class H264Encoder:
    """
    Read the H264/H265 profile and provide frames
    """
    def __init__(self, profile):
        """
        Format of the profile: <frame_id> <size> <psnr> <qp>
        """
        self.frame_id = 0
        self.MPEG_MIN_QP = 14
        self._load_video_profile(profile)

    def _load_video_profile(self, video_profile):
        """
        Input:
            video_profile: csv, format is <frame_id> <size> <psnr> <qp>
        """
        MPEG_MIN_QP = self.MPEG_MIN_QP
        self.profile = pd.read_csv(video_profile)
        freeze_psnr_row = self.profile.query("frame_id == -1")
        self.freeze_psnr = float(freeze_psnr_row["psnr"])
        self.profile = self.profile.query("qp > @MPEG_MIN_QP")
        self.nframes = max(self.profile["frame_id"])
        print("Load the information for {} frames".format(self.nframes))
        return self

    def _fit_size_for_frame(self, frame_id, size):
        """
        Input:
            frame_id: id of the frame
            size: the total size for a frame (NO FEC, NO SVC)
        Output:
            size: the real frame size
            psnr: the psnr of the frame
        """
        temp = self.profile.query("frame_id == @frame_id")
        tgt_size = size

        min_possible_size = min(temp["size"])
        worst_psnr = min(temp["psnr"])

        result_index = temp['size'].sub(tgt_size).abs().idxmin()
        temp = temp.query("index == @result_index")

        return float(temp["size"]), float(temp["psnr"])

    def get_next_frame(self, target_size)-> FrameInfo:
        """
        Input:
            target_size: the target size of a frame
        Returns:
            frame: the FrameInfo object, will be None if there is no more frames
        """
        if self.frame_id >= self.nframes:
            return None

        frame_id = self.frame_id
        self.frame_id += 1

        size, psnr = self._fit_size_for_frame(frame_id, target_size)
        return FrameInfo(frame_id, size, psnr)

class Autoencoder:
    """
    Read the autoencoder profile and provide frames
    """
    def __init__(self, profile):
        self.frame_id = 0
        self._load_video_profile(profile)

    def _load_video_profile(self, video_profile):
        """
        Input:
            video_profile: csv, format is <frame_id> <size> <psnr> <loss> <qp>
        """
        self.profile = pd.read_csv(video_profile)
        freeze_psnr_row = self.profile.query("frame_id == -1")
        self.freeze_psnr = float(freeze_psnr_row["psnr"])
        self.nframes = max(self.profile["frame_id"])
        print("Load the information for {} frames".format(self.nframes))
        return self

    def get_next_frame(self, target_sze) -> FrameInfo:
        """
        Input:
            target_size: the target size of the frame
        Output:
            frame: the frameinfo object
        """
        # TODO: here!

class FrameBuffer:
    """
    a buffer maintaining the frame send/recv status
    also provide the final PSNR v.s. latency calculation
    """
    def __init__(self):
        self.frames = {} # List[FrameInfo]

    def on_frame_sent(self, ts: float, frame: FrameInfo):
        """
        Note: it will call frame.on_sent(), so do not call it outside this class
        Input:
            ts: the timestamp of the frame being sent
            frame: the frame itself
            psnr: the psnr of that frame
        """
        fid = frame.frame_id
        if fid in self.frames:
            raise RuntimeError(f"Two frames share the same frame id {fid}")
        frame.on_sent(ts)
        self.frames[fid] = frame

    def on_packet_received(self, ts: float, pkt: PacketInfo):
        """
        Called when a packet is received
        ts: the timestamp when packet is received
        pkt: the packet
        """
        frame_id, frag_id = FrameInfo.TranslatePacketId(pkt.id)
        if frame_id not in self.frames:
            print(f"\033[33mWarning: Frame {frame_id} is not found in the frame buffer, skip!\033[0m")
            return
        self.frames[frame_id].on_recv(ts, pkt)

    def get_frames(self) -> List[FrameInfo]:
        return list(self.frames.values())

    def get_psnr_delay(self, skip_bad = True)-> pd.DataFrame:
        """
        Returns the psnr and delay in a DataFrame format
        The frame who are not received is skipped if skip_bad=True
        """
        data = [] # format <frame_id> <psnr> <delay> <size>
        for frame_id in self.frames.keys():
            frame = self.frames[frame_id]
            psnr = frame.psnr
            delay = frame.get_delay()
            size = frame.size
            if skip_bad and delay is None:
                continue
            data.append((frame_id, psnr, delay, size))

        ret = pd.DataFrame(data, columns=["frame_id", "psnr", "delay", "size"])
        return ret

#class TransmissionController:
#    """
#    Basic class for a transmission controller, provide some helper method for packet buffer and frame buffer
#    """

class NACKSender:
    """
    NACK transmission control, no extra traffic, using NACK for retransmission
    """
    def __init__(self, frame_rate: int):
        self.frame_buffer = FrameBuffer()
        self.pkt_buffer = PacketBuffer()
        self.rtx_buffer = PacketBuffer()
        self.target_bitrate_kbps = 100 # bitrate in KB per sec
        self.frame_rate = frame_rate # fps

    ''' Interfaces for derived classes to implement '''
    def on_new_frame(self, ts:float, frame: FrameInfo):
        """
        Called when a new frame generated by the encoder
        Packetize the new frame and update internal packet mapping
        Input:
            frame: the frame info object
        """
        self.frame_buffer.on_frame_sent(ts, frame)
        pkts = frame.packetize()
        for pkt in pkts:
            self.pkt_buffer.add_packet(pkt)

    def on_packet_feedback(self, ts: float, pkt_id: int, is_lost: bool, delay: float, recv_ts: float, size: int):
        """
        Called when a packet feedback is received
        Update the internal frame buffer status and handle retransmissions
        Input:
            ts: current timestamp
            pkt_id: the id of the received packet feedback
            is_lost: is the packet lost or not
            delay: the one-way-delay of the packet (sender->receiver)
            recv_ts: the time when packet is received
            size: the size of the packet in bytes
        """
        pkt = PacketInfo(pkt_id, size) # NOTE: the size of packet is unknown upon receiving, so use a default one
        if is_lost:
            # TODO: loss recovery
            self.rtx_buffer.add_packet(pkt)
        else:
            ''' add the packet into the frame buffer '''
            self.frame_buffer.on_packet_received(recv_ts, pkt)

    def on_target_bitrate_change(self, bitrate_byte_sec: int):
        """
        Called when target bitrate changes
        Input:
            target_bitrate: target bitrate in byte per sec
        """
        self.target_bitrate_kbps = bitrate_byte_sec / 1000 # convert to KB per sec

    def calculate_frame_size(self, interval: float) -> int:
        """
        Called when about to encode a new frame, returns a suitable size for the next frame in BYTES
        Input:
            interval: the interval between frames, in seconds
        Returns:
            the frame size in bytes
        """
        return self.target_bitrate_kbps * 1000 * interval

    def has_data(self, ts: float) -> bool:
        """
        ts: the current timestamp
        returns True if there is any data to send
        """
        return self.pkt_buffer.has_data() or self.rtx_buffer.has_data()

    def queue_len(self, ts: float) -> int:
        return self.pkt_buffer.queue_len() + self.rtx_buffer.queue_len()

    def get_packet(self) -> Tuple[int, int]:
        """
        returns packet id and size
        if no packets, return None, None
        """
        if self.rtx_buffer.has_data():
            pkt = self.rtx_buffer.get_packet()
            return pkt.id, pkt.size
        if self.pkt_buffer.has_data():
            pkt = self.pkt_buffer.get_packet()
            return pkt.id, pkt.size

        print("\033[33m Warning: called get_packet() but there is no packet in the buffer \033[0m")
        return None, None

    def get_frame_buffer(self):
        return self.frame_buffer

class VideoApplication(Application):
    def __init__(self, fps:int, init_bitrate: int, encoder:H264Encoder, trans:NACKSender):
        self.encoder = encoder
        self.trans = trans
        self.fps = fps

        self.curr_ts = 0
        self.last_frame_ts = -500

        self.trans.on_target_bitrate_change(init_bitrate)

    def tick(self, timestamp:float):
        """
        timestamp: the current timestamp
        """
        self.curr_ts = timestamp
        #print("[{:.4f}] in tick".format(self.curr_ts))
        if self.should_send_new_frame():
            target_size = self.trans.calculate_frame_size(self.curr_ts - self.last_frame_ts)
            frame = self.encoder.get_next_frame(target_size)
            if frame is None:
                return
            print("[{:.4f}] Generated a frame with size {} Bytes (target_rate={:.3f} real_rate={:.3f})".format(
                self.curr_ts, frame.size, self.trans.target_bitrate_kbps * 1000, frame.size / (self.curr_ts - self.last_frame_ts)))
            self.trans.on_new_frame(self.curr_ts, frame)
            self.last_frame_ts = self.curr_ts

    def should_send_new_frame(self) -> bool:
        return self.curr_ts - self.last_frame_ts > 1 / self.fps

    def queue_len(self, ts: float) -> bool:
        """
        ts: the current timestamp
        """
        self.tick(ts)
        return self.trans.queue_len(ts)

    def has_data(self, ts: float) -> bool:
        """
        ts: the current timestamp
        """
        self.tick(ts)
        return self.trans.has_data(ts)

    def get_packet(self)-> Tuple[int, int]:
        """
        returns: packet_id, packet_size in bytes
        """
        return self.trans.get_packet()

    def feedback(self, ts: float, packet_info: List[Tuple[int, bool, float, float, int, float]]):
        """
        Called when new packet is acked by the sender, triggered by CC
        Input:
            ts: current timestamp in second.
            packet_info: list of (pkt id, dropped, one way delay, recv
                    timestamp, pkt size (bytes), tgt bitrate (byte per sec))
        """
        for pkt in packet_info:
            tgt_bitrate = pkt[-1]
        self.trans.on_target_bitrate_change(tgt_bitrate)
        self.tick(ts)
        for info in packet_info:
            pktid, islost, delay, recv_ts, pkt_size, tgt_br = info
            self.trans.on_packet_feedback(ts, pktid, islost, delay, recv_ts, pkt_size)

    def get_summary(self) -> pd.DataFrame():
        return self.trans.get_frame_buffer().get_psnr_delay()

class VideoApplicationBuilder:
    def __init__(self, fps=25):
        self.fps = 25
        self.profile = None
        self.init_bitrate = 250 * 1000

    def set_init_bitrate(self, bitrate):
        """
        bitrate: in byte per sec
        """
        self.init_bitrate = bitrate

    def set_profile(self, profile):
        """
        profile: encoder profile
        """
        self.profile = profile
        return self

    def set_fps(self, fps):
        """
        fps: the frame rate in FPS
        """
        self.fps = fps
        return self

    def build(self) -> VideoApplication:
        assert self.fps is not None
        assert self.profile is not None, "need to set profile before build()"
        encoder = H264Encoder(self.profile)
        trans = NACKSender(self.fps)
        app = VideoApplication(self.fps, self.init_bitrate, encoder, trans)
        return app

''' ============ unit tests ============ '''
def test_h264_encoder():
    print("\033[32m===== start test_h264_encoder =====\033[0m")
    profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/test_mpeg.csv"
    enc = H264Encoder(profile)
    tot_sz = 0
    target_size = 30000
    for i in range(10):
        frame = enc.get_next_frame(target_size)
        fid = frame.frame_id
        sz = frame.size
        psnr = frame.psnr
        assert fid == i
        tot_sz += sz
    target_size *= 10
    assert tot_sz > 0.8 * target_size and tot_sz < 1.2 * target_size, f"Target size = {target_size}, but got {tot_sz}"
    print(f"target size = {target_size}, got size = {tot_sz}")
    print("\033[32m----- Passed test_h264_encoder -----\033[0m")

def test_frame_packet():
    print("\033[32m===== start test_frame_packet =====\033[0m")
    pkt = PacketInfo(1, 1000)
    assert pkt.id == 1
    assert pkt.size == 1000

    frame = FrameInfo(15, 12345, 30)
    pkt_infos = frame.packetize()
    assert len(pkt_infos) == 13

    assert frame.is_sent() == False
    frame.on_sent(1)
    assert frame.is_sent() == True

    tot_size = 0
    for pkt in pkt_infos:
        tot_size += pkt.size
        assert frame.all_received() == False
        frame.on_recv(tot_size, pkt)
    assert tot_size == 12345
    assert frame.all_received()
    assert frame.get_delay() == 12344
    print("\033[32m----- Passed test_frame_packet -----\033[0m")

def test_buffers():
    print("\033[32m===== start test_buffers =====\033[0m")
    ids = [1, 2, 3]
    sizes = [5000, 3000, 7000]
    psnrs = [35, 33, 37]

    ''' generate some frames '''
    ''' put them into frame buffer and packet buffer '''
    frame_buf = FrameBuffer()
    pkt_buf = PacketBuffer()
    for i in range(3):
        frame = FrameInfo(ids[i], sizes[i], psnrs[i])
        pkts = frame.packetize()
        for pkt in pkts:
            pkt_buf.add_packet(pkt)
        frame_buf.on_frame_sent(i, frame)

    assert pkt_buf.has_data()
    ''' mark them as received '''
    ts = 3
    while pkt_buf.has_data():
        pkt = pkt_buf.get_packet()
        frame_buf.on_packet_received(ts, pkt)
        ts += 1

    ''' get delay and psnr '''
    exp_recv_time = [7, 9, 15]
    for idx, frame in enumerate(frame_buf.get_frames()):
        assert exp_recv_time[idx] == frame.get_delay()

    df = frame_buf.get_psnr_delay()
    print(df)
    print("\033[32m----- Passed test_buffers -----\033[0m")

def test_nack_sender():
    print("\033[32m===== start test_nack_sender =====\033[0m")
    '''
    interfaces:
        on_new_frame, on_packet_feedback, on_target_bitrate_change
        calculate_frame_size, has_data, get_packet
    '''
    profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/test_mpeg.csv"
    encoder = H264Encoder(profile)
    fps = 25
    sender = NACKSender(fps)

    ''' test bitrate interface '''
    sender.on_target_bitrate_change(500 * 1000)     # 250 KB per sec --> 2000 Kbps
    target_size = sender.calculate_frame_size()
    assert target_size == 20000

    ''' test frame interface '''
    assert sender.has_data(0) == False
    frame = encoder.get_next_frame(target_size)
    sender.on_new_frame(0, frame)
    assert sender.has_data(0) == True

    ''' test get packets '''
    pkts = []
    while sender.has_data(0):
        pkts.append(sender.get_packet())
    print("got {} pkts".format(len(pkts)))
    assert sender.has_data(0) == False

    delay = 50
    for id, sz in pkts[:-1]:
        sender.on_packet_feedback(delay, id, False, delay, delay, sz)
    # now the frame is not all received
    print(sender.get_frame_buffer().get_psnr_delay(skip_bad=False))
    id, sz = pkts[-1]
    sender.on_packet_feedback(delay, id, True, delay, delay, sz)
    assert sender.has_data(0) == True # have RTX packet
    pkt = sender.get_packet()
    assert pkt is not None
    id, sz = pkts[-1]
    sender.on_packet_feedback(delay, id, False, delay, 2 * delay, sz)
    assert sender.has_data(0) == False # sent the RTX packet
    # now we should have the delay
    print(sender.get_frame_buffer().get_psnr_delay(skip_bad=False))
    print("\033[32m----- Passed test_nack_sender -----\033[0m")


def test_app():
    print("\033[32m===== start test_app =====\033[0m")
    builder = VideoApplicationBuilder()
    profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/test_mpeg.csv"
    tgt_br = 250 * 1000
    builder.set_profile(profile).set_fps(25).set_init_bitrate(tgt_br)
    app = builder.build()

    assert app.has_data(0) == True
    ts = 0.01
    while app.has_data(0):
        id, sz = app.get_packet()
        lost = np.random.uniform(0, 1) < 0.2
        app.feedback(ts, [(id, lost, ts, ts, sz, tgt_br)])
        print(f"packet id={id}, size={sz}, islost={lost}")

    print("\033[32m----- Passed test_app -----\033[0m")


if __name__ == "__main__":
    test_h264_encoder()
    test_frame_packet()
    test_buffers()
    test_nack_sender()
    test_app()
