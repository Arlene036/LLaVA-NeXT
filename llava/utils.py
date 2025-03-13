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
import traceback
import imageio

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():
  def load(self, file_handle):
        file_handle.seek(16 * 4, 1)
        file_handle.seek(8, 1)
        file_handle.seek(8, 1)
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        file_handle.seek(self.depth_size_bytes, 1)

  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)
  
class SensReader:

    def __init__(self, filename):
        try:
            self._file = open(filename, 'rb')
            file_size = os.path.getsize(filename)
            
            version_data = self._file.read(4)
            version = struct.unpack('I', version_data)[0]
            
            strlen_data = self._file.read(8)
            strlen = struct.unpack('Q', strlen_data)[0]
            
            self._file.read(strlen)
           
            camera_params_size = 16*4*4
            self._file.read(camera_params_size)
          
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', self._file.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', self._file.read(4))[0]]
        
            width_data = self._file.read(4)
            self.color_width = struct.unpack('I', width_data)[0]
            
            height_data = self._file.read(4)
            self.color_height = struct.unpack('I', height_data)[0]
            
            depth_size_data = self._file.read(8 + 4)
      
            num_frames_data = self._file.read(8)
            self.num_frames = struct.unpack('Q', num_frames_data)[0]
            
            self.frames = []
            for i in range(self.num_frames):
                frame = RGBDFrame()
                frame.load(self._file)
                self.frames.append(frame)
            
        except Exception as e:
            print(f"error when init SensReader: {e}")
            traceback.print_exc()
            if hasattr(self, '_file') and self._file:
                self._file.close()
            raise


    def get_batch(self, frame_indices):
        frames = []
        for idx in frame_indices:
            if idx < 0 or idx >= len(self.frames):
                raise IndexError(f"frame index out of range: {idx}, total frames: {len(self.frames)}")
            
            frame = self.frames[idx]
            
            try:
                color_array = frame.decompress_color(self.color_compression_type)
                frames.append(color_array)
            except Exception as e:
                print(f"error when decompress frame {idx}: {e}")
                empty_frame = np.zeros((self.color_height, self.color_width, 3), dtype=np.uint8)
                frames.append(empty_frame)
        
        if frames:
            return np.stack(frames, axis=0)
        else:
            return np.zeros((0, self.color_height, self.color_width, 3), dtype=np.uint8)
        
    def __del__(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()


def process_video_with_sens(sens_file, data_args):
    try:
        video_fps = data_args.video_fps
        if not os.path.exists(sens_file):
            raise FileNotFoundError(f"file not found: {sens_file}")
        
        file_size = os.path.getsize(sens_file)
        
        reader = SensReader(sens_file)
        total_frame_num = reader.num_frames
        
        if total_frame_num == 0:
            raise ValueError("no frames found")
        
        video_time = reader.num_frames / video_fps
        
        avg_fps = data_args.video_fps
        avg_fps = round(avg_fps)
        frame_idx = list(range(0, total_frame_num, avg_fps))
        frame_time = [i/avg_fps for i in frame_idx]
        
        if hasattr(data_args, 'frames_upbound') and data_args.frames_upbound > 0:
            force_sample = getattr(data_args, 'force_sample', False)
            if len(frame_idx) > data_args.frames_upbound or force_sample:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i/video_fps for i in frame_idx]

        video = reader.get_batch(frame_idx)
        
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        
        num_frames_to_sample = len(frame_idx)
        
        frames = [Image.fromarray(video[i]) for i in range(video.shape[0])]
        
        return frames, video_time, frame_time_str, num_frames_to_sample
    
    except Exception as e:
        print(f"error when process .sens file: {e}")
        traceback.print_exc()
        return [], 0.0, "", 0