import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"



import os
import numpy as np
from PIL import Image
import struct
import zlib
import io

class SensReader:
    """简化版 ScanNet .sens 文件读取器，专注于快速提取 RGB 帧"""
    
    def __init__(self, filename):
        """初始化 SensReader
        
        参数:
            filename: .sens 文件路径
        """
        self._file = open(filename, 'rb')
        self._file.read(4)
        
        strlen = struct.unpack('Q', self._file.read(8))[0]
        self._file.read(strlen)
        
        self._file.read(16*4*4)
        self._file.read(8)
        
        self.color_width = struct.unpack('I', self._file.read(4))[0]
        self.color_height = struct.unpack('I', self._file.read(4))[0]
        
        self._file.read(8 + 4)

        self.num_frames = struct.unpack('Q', self._file.read(8))[0]
        
        self._frame_indices = []
        

        for _ in range(self.num_frames):
            self._file.seek(16*4, 1)
            timestamp_color = struct.unpack('Q', self._file.read(8))[0]
            self._file.read(16)
            color_size_bytes = struct.unpack('Q', self._file.read(8))[0]
            depth_size_bytes = struct.unpack('Q', self._file.read(8))[0]
            
            current_pos = self._file.tell()
            self._frame_indices.append((current_pos, color_size_bytes))
            self._timestamps.append(timestamp_color)
            
            self._file.seek(color_size_bytes + depth_size_bytes, 1)
    
    def get_rgb_frame(self, frame_idx):
        """获取指定索引的 RGB 帧
        
        参数:
            frame_idx: 帧索引
            
        返回:
            PIL.Image: RGB 图像
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"帧索引超出范围: {frame_idx}, 总帧数: {self.num_frames}")
        
        pos, color_size = self._frame_indices[frame_idx]
        self._file.seek(pos)
        
        color_data = self._file.read(color_size)
        
        try:
            color_image = Image.open(io.BytesIO(color_data))
            return color_image
        except:
            try:
                color_array = np.frombuffer(color_data, dtype=np.uint8).reshape(self.color_height, self.color_width, 3)
                color_image = Image.fromarray(color_array)
                return color_image
            except:
                print(f"无法解析帧 {frame_idx} 的颜色数据")
                return Image.new('RGB', (self.color_width, self.color_height), (0, 0, 0))
    
    def get_timestamp(self, frame_idx):
        """获取指定索引帧的时间戳
        
        参数:
            frame_idx: 帧索引
            
        返回:
            int: 时间戳（微秒）
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"帧索引超出范围: {frame_idx}, 总帧数: {self.num_frames}")
        
        return self._timestamps[frame_idx]
    
    def get_video_duration(self):
        """计算视频总时长
        
        返回:
            float: 视频总时长（秒）
        """
        if self.num_frames <= 1:
            return 0.0
        
        first_timestamp = self._timestamps[0]
        last_timestamp = self._timestamps[-1]
        duration_us = last_timestamp - first_timestamp
        duration_s = duration_us / 1000000.0
        
        if duration_s <= 0:
            return self.num_frames / 30.0
        
        return duration_s
    
    def __del__(self):
        """关闭文件"""
        if hasattr(self, '_file') and self._file:
            self._file.close()


def process_video_with_sens(sens_file, data_args):
    """
    快速处理 ScanNet 的 .sens 文件，提取 RGB 帧序列
    
    参数:
        sens_file: .sens 文件路径
        data_args: 数据参数，包含 video_fps 和 frames_upbound
        
    返回:
        video: 帧列表，每个元素是 PIL.Image
        video_time: 视频总时长（秒）
        frame_time: 每一帧的时间戳字符串
        num_frames_to_sample: 采样的帧数
    """
    try:
        reader = SensReader(sens_file)
        total_frame_num = reader.num_frames
        
        video_time = reader.get_video_duration()
        original_fps = total_frame_num / video_time if video_time > 0 else 30.0

        if data_args.frames_upbound > 0:
            num_frames_to_sample = min(data_args.frames_upbound, total_frame_num)
            frame_indices = np.linspace(0, total_frame_num - 1, num_frames_to_sample, dtype=int)
        else:
            sample_interval = max(1, round(original_fps / (data_args.video_fps or 1.0)))
            frame_indices = np.arange(0, total_frame_num, sample_interval)
            num_frames_to_sample = len(frame_indices)
        
        frames = []
        frame_times = []
        
        first_timestamp = reader.get_timestamp(0) / 1000000.0  
        
        for idx in frame_indices:
            frames.append(reader.get_rgb_frame(int(idx)))
            timestamp = reader.get_timestamp(int(idx)) / 1000000.0
            relative_time = timestamp - first_timestamp
            frame_times.append(relative_time)
        
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_times])
        
        return frames, video_time, frame_time_str, num_frames_to_sample
    
    except Exception as e:
        print(f"处理 .sens 文件时出错: {e}")
        return [], 0.0, "", 0