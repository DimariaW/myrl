from myrl.utils import set_process_logger
from myrl.utils import wrap_traceback
import logging


@wrap_traceback(handle=open("./log/test.txt", "a+"))
def test_logger():
    set_process_logger(file_path="./log/test.txt")
    logging.info("test info")
    logging.debug("test debug")
    raise RuntimeError("test error")


