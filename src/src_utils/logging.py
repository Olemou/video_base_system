import logging
import os
import sys
import torch
LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name=None, force=False, log_dir=None, log_file="training_log.txt"):
    # Set default directory to ~/log if not provided
    if log_dir is None:
        log_dir = os.path.expanduser('~/log')
    
    # Ensure the log directory exists, create it if it doesn't
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Define the full log file path
    log_file_path = os.path.join(log_dir, log_file)
    
    # Set up the logger
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    # Formatter for the log messages
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # StreamHandler for stdout (console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler for writing logs to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to file {log_file_path}...")

    return logger


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        # -- print headers
        with open(self.fname, mode) as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)
