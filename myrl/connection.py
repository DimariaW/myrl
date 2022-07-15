# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
import io
import time
import struct
import socket
import pickle
import threading
import queue
import multiprocessing as mp
import traceback
from typing import Callable, Iterable
import myrl.utils as utils
import time
import logging


def send_recv(conn, sdata):
    conn.send(sdata)
    rdata = conn.recv()
    return rdata


class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fileno(self):
        return self.conn.fileno()

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            if len(chunk) == 0:
                raise ConnectionResetError
            size -= len(chunk)
            buf.write(chunk)
        return buf

    def recv(self):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf):
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg):
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack("!i", n)
        if n > 16384:
            chunks = [header, buf]
        elif n > 0:
            chunks = [header + buf]
        else:
            chunks = [header]
        for chunk in chunks:
            self._send(chunk)


def open_socket_connection(port, reuse=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR,
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
    )
    hostname = socket.gethostbyname(socket.gethostname())
    sock.bind((hostname, int(port)))
    logging.info(f"successfully bind {hostname}:{port}")
    return sock


def accept_socket_connection(sock):
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


def listen_socket_connections(n, port):
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def connect_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, int(port)))
    except ConnectionRefusedError as exception:
        logging.info('failed to connect %s %d' % (host, port))
        raise exception
    return PickledConnection(sock)


def accept_socket_connections(port, timeout=None, maxsize=1024):
    sock = open_socket_connection(port)
    sock.listen(maxsize)
    sock.settimeout(timeout)
    cnt = 0
    while cnt < maxsize:
        conn = accept_socket_connection(sock)
        if conn is not None:
            cnt += 1
        yield conn


def open_multiprocessing_connections(num_process, target, args_func):
    # open connections
    s_conns, g_conns = [], []
    for _ in range(num_process):
        conn0, conn1 = mp.connection.Pipe(duplex=True)
        s_conns.append(conn0)
        g_conns.append(conn1)

    # open workers
    for i, conn in enumerate(g_conns):
        mp.Process(target=target, args=args_func(i, conn)).start()
        conn.close()

    return s_conns


#%%


"""
the cell of MultiProcessJobExecutor
"""


def wrapped_func(func: Callable, conn, logger_file_path=None):
    try:
        utils.set_process_logger(file_path=logger_file_path)
        total_sent = 0
        while True:
            beg = time.time()
            data = conn.recv()
            data = func(data)
            time.sleep(max(0.01, 0.1 - (time.time()-beg)))
            conn.send((data, 1))
            total_sent += 1
            logging.debug(f"successfully send data counts is : {total_sent}")
    except Exception:
        traceback.print_exc(file=open(logger_file_path, "a") if logger_file_path is not None else None)
        raise


class MultiProcessJobExecutors:
    def __init__(self,
                 func: Callable,
                 send_generator: Iterable,
                 num: int,
                 postprocess: Callable = None,
                 buffer_length: int = 8,
                 num_receivers: int = 1,
                 name_prefix: str = None,
                 logger_file_path: str = None
                 ):
        """

        :param func:
        :param send_generator:
        :param num:
        :param postprocess:
        :param buffer_length:
        :param num_receivers:
        :param logger_file_path:

        launch num process, each process return func(next(send_generator)) to a queue,
        the main process can use queue.get() to get the results,

        the buffer_length is the total data can be sent ahead of receiving.
        the num_receivers control how many receiver thread can be launched.

        each job executors have a process name: f"{name_prefix}_{i}"
        the logging info will be written in to logger_file_path.
        """
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.buffer_length = buffer_length
        self.num_receivers = num_receivers
        self.conns = []
        self.send_cnt = {}
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=8)
        self.threads = []
        self.name_prefix = name_prefix
        self.logger_file_path = logger_file_path

        for i in range(num):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(name=f"{name_prefix}-{i}",
                       target=wrapped_func,
                       args=(func, conn1, logger_file_path), daemon=True).start()
            conn1.close()
            self.conns.append(conn0)
            self.send_cnt[conn0] = 0

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.output_queue.get()

    def start(self):
        self.threads.append(threading.Thread(name="sender thread", target=self._sender, daemon=True))
        for i in range(self.num_receivers):
            self.threads.append(threading.Thread(name=f"receiver thread {i}",
                                                 target=self._receiver,
                                                 args=(i,),
                                                 daemon=True))
        for thread in self.threads:
            thread.start()

    def _sender(self):
        logging.info("start send data")
        while not self.shutdown_flag:
            total_send_cnt = 0
            for conn, cnt in self.send_cnt.items():
                if cnt < self.buffer_length:
                    conn.send(next(self.send_generator))
                    self.lock.acquire()
                    self.send_cnt[conn] += 1
                    self.lock.release()
                    total_send_cnt += 1
            if total_send_cnt == 0:
                time.sleep(0.01)
        logging.info('finished sender')

    def _receiver(self, index):
        logging.info('start receiver %d' % index)
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = mp.connection.wait(conns)
            for conn in tmp_conns:
                data, cnt = conn.recv()

                if self.postprocess is not None:
                    data = self.postprocess(data)

                while True:
                    """
                    只有成功put 数据，才修改send cnt
                    """
                    try:
                        self.output_queue.put(data, timeout=0.1)
                        self.lock.acquire()
                        self.send_cnt[conn] -= cnt
                        self.lock.release()
                        break
                    except queue.Full:
                        logging.debug("output_queue is full, the bottleneck is the speed of learner consume batch")
        logging.info('end receiver %d' % index)


#%%


class QueueCommunicator:
    def __init__(self, conns=[]):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = set()
        for conn in conns:
            self.add_connection(conn)
        threading.Thread(target=self._send_thread, daemon=True).start()
        threading.Thread(target=self._recv_thread, daemon=True).start()

    def connection_count(self):
        return len(self.conns)

    def recv(self, timeout=None):
        return self.input_queue.get(timeout=timeout)

    def send(self, conn, send_data):
        self.output_queue.put((conn, send_data))

    def add_connection(self, conn):
        self.conns.add(conn)

    def disconnect(self, conn):
        self.conns.discard(conn)
        logging.info(f'disconnected one connection, current connection num is {self.connection_count()}')

    def _send_thread(self):
        while True:
            conn, send_data = self.output_queue.get()
            try:
                conn.send(send_data)
            except ConnectionResetError:
                self.disconnect(conn)
            except BrokenPipeError:
                self.disconnect(conn)

    def _recv_thread(self):
        while True:
            conns = mp.connection.wait(self.conns, timeout=0.3)
            for conn in conns:
                try:
                    recv_data = conn.recv()
                except ConnectionResetError:
                    self.disconnect(conn)
                    continue
                except EOFError:
                    self.disconnect(conn)
                    continue

                while True:
                    try:
                        self.input_queue.put((conn, recv_data), timeout=0.3)
                        break
                    except queue.Full:
                        logging.critical("this process cannot consume some manny actor, the message queue is full")



