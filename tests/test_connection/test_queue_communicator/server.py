import myrl.utils as utils
import myrl.connection as connection

import logging

if __name__ == "__main__":
    utils.set_process_logger()
    communicator = connection.QueueCommunicator(port=11111)
    communicator.run()
    while True:
        conn, msg = communicator.recv()
        logging.info(msg)
        communicator.send(conn, f"received{communicator.connection_count()}")
