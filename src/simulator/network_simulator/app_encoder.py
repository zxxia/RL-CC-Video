import numpy as np
import pandas as pd
from .app_basic import FrameInfo, AEFrameInfo

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

    def _fit_size_for_frame(self, frame_id, size):
        """
        Input:
            frame_id: id of the frame
            size: the total size for a frame (NO FEC, NO SVC)
        Output:
            size: the real frame size
            psnr: the psnr of the frame
        """
        temp = self.profile.query("frame_id == @frame_id and loss == 0")
        tgt_size = size

        result_index = temp['size'].sub(tgt_size).abs().idxmin()
        temp = temp.query("index == @result_index")

        return float(temp["size"]), float(temp["psnr"])

    def query_frame_with_size(self, frame_id, size, loss = 0) -> float:
        """
        Input:
            frame_id: the id of the frame
            size: the exact size of the frame
            loss: the real loss rate of that frame
        Output:
            psnr: the estimated psnr of that frame
        """
        temp = self.profile.query("frame_id == @frame_id and size == @size")
        psnr = np.interp(loss, temp["loss"], temp["psnr"])
        return psnr

    def get_next_frame(self, target_size) -> AEFrameInfo:
        """
        Input:
            target_size: the target size of the frame
        Output:
            frame: the frameinfo object
        """
        if self.frame_id >= self.nframes:
            return None

        frame_id = self.frame_id
        self.frame_id += 1

        size, psnr = self._fit_size_for_frame(frame_id, target_size)
        return AEFrameInfo(frame_id, size, psnr, self)
