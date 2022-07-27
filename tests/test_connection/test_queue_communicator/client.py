import myrl.connection as connection
import myrl.utils as utils

import numpy as np
import logging

if __name__ == "__main__":
    utils.set_process_logger()
    conn = connection.connect_socket_connection("172.18.237.51", 11111)
    logging.info(connection.send_recv(conn, np.random.randn(16, 76, 92)))
