import logging
import os
import sys

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