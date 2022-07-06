import logging
from functools import wraps
import traceback
import sys


allowed_levels = (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)


def set_process_logger(name=None, stdout_level=logging.INFO, file_path=None, file_level=logging.DEBUG):
    if stdout_level not in allowed_levels or file_level not in allowed_levels:
        raise ValueError(" level is not allowed")

    logger = logging.getLogger(name=name)

    # logger level need to be debug level
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="[pid: %(process)d, pname: %(processName)s], "
                                      "[tid: %(thread)d, tname: %(threadName)s], "
                                      "[%(asctime)s]: %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")

    s_handler = logging.StreamHandler()
    s_handler.setLevel(stdout_level)
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(file_level)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


def wrap_traceback(handle=sys.stderr):
    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except:
                traceback.print_exc(file=handle)
                handle.flush()
        return wrapped_func
    return wrap
