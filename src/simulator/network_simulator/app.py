import pandas as pd
import numpy as np
from typing import List, Tuple
from .app_basic import FrameInfo, AEFrameInfo, PacketInfo, PacketBuffer
from .app_encoder import H264Encoder, Autoencoder
from .app_transport import FrameBuffer, NACKSender, SimpleSender


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
        #if self.curr_ts != timestamp:
        #    print("[{:.4f}] in tick".format(self.curr_ts))
        self.curr_ts = timestamp
        if self.should_send_new_frame():
            target_size = self.trans.calculate_frame_size(self.curr_ts - self.last_frame_ts)
            frame = self.encoder.get_next_frame(target_size)
            if frame is None:
                return
            print("[{:.4f}] Generated a frame [{}] with size {} Bytes (target_rate={:.3f} real_rate={:.3f})".format(
                self.curr_ts, frame.frame_id, frame.size, self.trans.target_bitrate_kbps * 1000, frame.size / (self.curr_ts - self.last_frame_ts)))
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

        # preset values
        self.MPEG_NACK = 1
        self.AE_SIMPLE = 2
        self.presets = [
                    self.MPEG_NACK,
                    self.AE_SIMPLE,
                ]
        self.preset = self.MPEG_NACK

    def set_init_bitrate(self, bitrate):
        """
        bitrate: in byte per sec
        """
        self.init_bitrate = bitrate
        return self

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

    def set_preset(self, preset):
        if preset not in self.presets:
            raise RuntimeError("VideoApplicationBuilder: Preset not found!")
        self.preset = preset
        return self

    def build(self) -> VideoApplication:
        assert self.fps is not None
        assert self.profile is not None, "need to set profile before build()"
        assert self.init_bitrate is not None
        if self.preset == self.MPEG_NACK:
            encoder = H264Encoder(self.profile)
            trans = NACKSender(self.fps)
        elif self.preset == self.AE_SIMPLE:
            encoder = Autoencoder(self.profile)
            trans = SimpleSender(self.fps) 
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

def test_senders(sender:SimpleSender):
    print("\033[32m===== start test_nack_sender =====\033[0m")
    '''
    interfaces:
        on_new_frame, on_packet_feedback, on_target_bitrate_change
        calculate_frame_size, has_data, get_packet
    '''
    profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/test_mpeg.csv"
    encoder = H264Encoder(profile)
    fps = sender.frame_rate
    #sender = NACKSender(fps)

    ''' test bitrate interface '''
    sender.on_target_bitrate_change(500 * 1000)     # 250 KB per sec --> 2000 Kbps
    target_size = sender.calculate_frame_size(1/fps)
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
    if isinstance(sender, NACKSender):
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

def test_ae_frame():
    print("\033[32m===== start test_ae_frame =====\033[0m")
    profile = "/datamirror/yihua98/projects/autoencoder_testbed/sim_db/merged-loss/-00nar1nEPc_000033_000043-ae.csv"
    ae_encoder = Autoencoder(profile)

    frame = ae_encoder.get_next_frame(12000)
    assert isinstance(frame, AEFrameInfo)
    print("Original frame:", frame.frame_id, frame.size, frame.psnr)
    pkts = frame.packetize()
    frame.on_sent(0)
    assert frame.get_delay() is None
    n_lost = 0
    for idx, pkt in enumerate(pkts):
        if np.random.uniform(0, 1) < 0.3:
            n_lost += 1
            continue
        frame.on_recv(idx, pkt)
        assert frame.get_delay() == idx, "Got delay = {} but expected {}".format(frame.get_delay(), idx)
    print("Lost {} out of {} packets".format(n_lost, len(pkts)))
    print("Estimated psnr:", frame.get_psnr())
    print("\033[32m----- Passed test_ae_frame -----\033[0m")


def test_main():
    test_h264_encoder()
    test_frame_packet()
    test_buffers()
    #test_nack_sender()
    test_senders(SimpleSender(25))
    test_senders(NACKSender(25))
    test_app()
    test_ae_frame()
    exit(0)
