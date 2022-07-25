import numpy as np
import torch
import logging
from functools import wraps
import traceback
import sys
import os
from typing import Union, List, Dict, Tuple


allowed_levels = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)


class DirFilter(logging.Filter):
    def __init__(self, dir_head):
        super().__init__()
        self.dir_head = dir_head

    def filter(self, record: logging.LogRecord) -> bool:
        temp = os.path.commonprefix([record.pathname, self.dir_head])
        if os.path.exists(temp) and os.path.samefile(temp, self.dir_head):
            return True
        else:
            return False


def set_process_logger(name=None, stdout_level=logging.INFO, file_path=None, file_level=logging.DEBUG,
                       starts_with=None):
    """
    note: name is usually None, representing the root logger.
    when using fork, the logger is transferred to other process.
    so we prefer to use spawn method.
    starts_with means only log the message with file is starts from this dir
    """
    if stdout_level not in allowed_levels or file_level not in allowed_levels:
        raise ValueError(" level is not allowed")

    logger = logging.getLogger(name=name)

    # logger level need to be debug level
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="[pid: %(process)d, pname: %(processName)s], "
                                      "[tid: %(thread)d, tname: %(threadName)s], "
                                      "[%(asctime)s], "
                                      "[%(pathname)s-%(lineno)d]: "
                                      "%(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    s_handler = logging.StreamHandler()
    s_handler.setLevel(stdout_level)
    s_handler.setFormatter(formatter)
    if starts_with is not None:
        s_handler.addFilter(DirFilter(starts_with))
    logger.addHandler(s_handler)

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(file_level)
        f_handler.setFormatter(formatter)
        if starts_with is not None:
            f_handler.addFilter(DirFilter(starts_with))
        logger.addHandler(f_handler)
    return logger


def wrap_traceback(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            logging.error(traceback.format_exc())
            raise
    return wrapped_func


def to_tensor(x: Union[List, Dict, Tuple, np.ndarray, torch.Tensor], unsqueeze=None, device=torch.device("cpu")):
    if isinstance(x, (list, tuple)):
        return type(x)(to_tensor(xx, unsqueeze, device) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, to_tensor(xx, unsqueeze, device)) for key, xx in x.items())
    elif isinstance(x, np.ndarray):
        if x.dtype in [np.int32, np.int64]:
            t = torch.from_numpy(x).type(torch.int64).to(device)
        else:
            t = torch.from_numpy(x).type(torch.float32).to(device)
        return t if unsqueeze is None else t.unsqueeze(unsqueeze)
    elif isinstance(x, torch.Tensor):
        if x.dtype in [torch.int32, torch.int64]:
            t = x.type(torch.int64).to(device)
        else:
            t = x.type(torch.float32).to(device)
        return t if unsqueeze is None else t.unsqueeze(unsqueeze)
    else:
        raise NotImplementedError(f"do not support convert type: {type(x)} to tensor")


def batchify(x: Union[List, Tuple], unsqueeze=None) -> Union[List, Tuple, Dict, np.ndarray]:
    if isinstance(x[0], (list, tuple)):
        temp = []
        for xx in zip(*x):
            temp.append(batchify(xx, unsqueeze))
        return type(x[0])(temp)

    elif isinstance(x[0], dict):
        temp = {}
        for key in x[0].keys():
            values = [xx[key] for xx in x]
            temp[key] = batchify(values, unsqueeze)
        return temp

    elif isinstance(x[0], np.ndarray):
        if unsqueeze is not None:
            return np.stack(x, axis=0)
        else:
            return np.concatenate(x, axis=0)

    else:
        if unsqueeze is None:
            raise ValueError(f"there are unbatchified dtype {type(x[0])}")
        else:
            return np.array(x)
