# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
import io
import os
import time
import struct
import socket
import pickle
import threading
import queue
import multiprocessing as mp

from typing import Callable, Iterator, Tuple, Any, Union
import logging

import torch

import myrl.utils as utils


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


def send_recv(conn: PickledConnection, sdata: Tuple[str, Any]) -> Tuple[str, Any]:  # sdata (cmd, args or data)
    conn.send(sdata)
    rdata = conn.recv()
    return rdata


def open_socket_connection(port, reuse=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR,
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
    )
    ip = socket.gethostbyname(socket.gethostname())
    sock.bind((ip, int(port)))
    logging.info(f"successfully bind {ip}:{port}")
    return sock


def listen_socket_connections(n, port):
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def accept_socket_connection(sock):
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


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


def connect_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, int(port)))
    except ConnectionRefusedError as exception:
        logging.info('failed to connect %s %d' % (host, port))
        raise exception
    return PickledConnection(sock)


class QueueCommunicatorBase:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = set()
        threading.Thread(target=self._send_thread, daemon=True).start()
        threading.Thread(target=self._recv_thread, daemon=True).start()

    def connection_count(self):
        return len(self.conns)

    def recv(self, timeout=None) -> Tuple[PickledConnection, Tuple[str, Any]]:
        return self.input_queue.get(timeout=timeout)

    def send(self, conn: PickledConnection, send_data: Tuple[str, Any]):  # send_data (cmd, args or data)
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


class QueueCommunicator(QueueCommunicatorBase):
    def __init__(self, port: int, num_client=None):
        """
        :param port: 指定服务器端口
        :param num_client: 指定连接的actor个数, 若异步模式此参数无意义, 若同步模式此参数表示需要等待actor_num个连接
        """
        super().__init__()
        self.port = port
        self.num_client = num_client

    def run(self):
        """
        异步模式
        """
        def worker_server(port):
            logging.info('preparing bind port: %d' % port)
            conn_acceptor = accept_socket_connections(port=port, maxsize=9999)
            while True:
                conn = next(conn_acceptor)
                self.add_connection(conn)
                logging.info(f"total connection count now is {self.connection_count()}")

        threading.Thread(name="add_connection", target=worker_server, args=(self.port,), daemon=True).start()

    def run_sync(self):
        """
        同步，堵塞直到所有actor建立连接
        """
        if self.num_client is None:
            raise ValueError("sync version requires known number of client")

        logging.info('preparing bind port: %d' % self.port)
        conn_acceptor = accept_socket_connections(port=self.port, maxsize=self.num_client)
        while True:
            try:
                conn = next(conn_acceptor)
                self.add_connection(conn)
                logging.info(f"total connection count now is {self.connection_count()}")
            except StopIteration:
                break

#%%


def send_with_stop_flag(queue_sender: mp.Queue, is_stop: bool, data: Any,  # model_id
                        block: bool = True, timeout: float = None):
    queue_sender.put((is_stop, data), block=block, timeout=timeout)


class Receiver:
    """
    mp.Queue通信时，sender端会发送一个is_stop标志位，通过对接收端的mp.Queue进行简单包装，可以在确保没有数据的情况下再抛出异常
    a simple wrapper on mp.Queue, that  when raise a queue.Empty exception, it means the all sender have sent all data
    """
    def __init__(self, queue_receiver: mp.Queue, num_sender: int = -1, postprocess=None):
        self.queue_receiver = queue_receiver
        self.num_sender = num_sender
        self.postprocess = postprocess

        self.stopped = False
        self.stopped_num = 0

    def recv(self):
        while True:
            try:
                logging.debug(f"current queue size is {self.queue_receiver.qsize()} ")
                is_stop, data = self.queue_receiver.get(timeout=0.1)

                if is_stop:
                    self.stopped_num += 1
                    if self.stopped_num == self.num_sender:
                        logging.debug("successfully receive all processed data!")
                        self.stopped = True
                    continue

                if self.postprocess is not None:
                    data = self.postprocess(data)

                return data

            except queue.Empty:
                if self.stopped:
                    raise


@utils.wrap_traceback
def wrapped_func(func: Callable, queue_sender: mp.Queue, queue_receiver: mp.Queue,
                 # 进程的logger信息
                 logger_file_path: str = None, file_level=logging.DEBUG, starts_with=None,
                 # 发送完数据后等待时间,
                 # 当发送torch.Tensor且 sharing strategy 为 file descriptor (linux 平台默认) 时，
                 # 必须等待一段时间等接收端完全接收
                 waiting_time: int = 0
                 ):
    utils.set_process_logger(file_path=logger_file_path, file_level=file_level, starts_with=starts_with)
    logging.info("start processing !")
    num_processed_data = 0
    while True:
        is_stop, data = queue_sender.get()
        if is_stop:
            logging.info("the sender is closed, this process is going to close!")
            send_with_stop_flag(queue_receiver, is_stop, data)
            break

        processed_data = func(data)
        while True:
            try:
                send_with_stop_flag(queue_receiver, is_stop, processed_data, timeout=0.3)
                num_processed_data += 1
                logging.debug(f"successfully processed data count {num_processed_data}!")
                break
            except queue.Full:
                logging.debug(" the receive queue is full !")
    time.sleep(waiting_time)


class MultiProcessJobExecutors:
    def __init__(self,
                 # task args
                 func: Callable,
                 send_generator: Iterator,
                 # 并行进程个数，与每个进程可以提前send的data个数
                 num: int,
                 buffer_length: int = 1,
                 # 是否外部传入queue_receiver, 如果传入的话，就不开启接收进程与后处理
                 queue_receiver: mp.Queue = None,
                 post_process: Callable = None,
                 # logger args
                 name_prefix: str = "",
                 logger_file_dir: str = None,
                 file_level=logging.DEBUG,
                 starts_with=None,
                 waiting_time=0
                 ):
        """
        launch num process each process return func(next(send_generator)) to a queue,
        the main process can use self.recv() to get the results,

        the buffer_length is the total data can be sent ahead of receiving.
        the num control how many receiver thread can be launched.

        each job executors have a process name: f"{name_prefix}_{i}"
        the logging info will be written in to logger_file_path.
        """
        self.send_generator = send_generator

        self.num = num

        self.start_receiver = True if queue_receiver is None else False
        self.queue_receiver = mp.Queue(maxsize=8) if self.start_receiver else queue_receiver
        self.output_queue = queue.Queue(maxsize=8) if self.start_receiver else None
        self.post_process = post_process

        self.queue_senders = []

        for i in range(num):
            queue_sender = mp.Queue(maxsize=buffer_length)

            if logger_file_dir is not None:
                logger_file_path = os.path.join(logger_file_dir, f'{name_prefix}-{i}.txt')
            else:
                logger_file_path = None

            mp.Process(name=f"{name_prefix}-{i}",
                       target=wrapped_func,
                       args=(func, queue_sender, self.queue_receiver,
                             logger_file_path, file_level, starts_with, waiting_time), daemon=True).start()

            self.queue_senders.append(queue_sender)

        self.threads = []
        self.stopped = False

    def recv(self):
        while True:
            try:
                data = self.output_queue.get(timeout=0.1)
                return data
            except queue.Empty:
                if self.stopped:
                    raise
                else:
                    pass

    def start(self):
        self.threads.append(threading.Thread(name="sender thread", target=self._sender, daemon=True))
        if self.start_receiver:
            self.threads.append(threading.Thread(name=f"receiver thread",
                                                 target=self._receiver,
                                                 daemon=True))
        for thread in self.threads:
            thread.start()

    def _sender(self):
        logging.info("start send data")
        try:
            while True:
                for queue_sender in self.queue_senders:
                    if not queue_sender.full():
                        send_with_stop_flag(queue_sender, False, next(self.send_generator))
        except StopIteration:
            for queue_sender in self.queue_senders:
                send_with_stop_flag(queue_sender, True, None)
            logging.info("successfully send all data!")

    def _receiver(self):
        logging.info('start receiver')
        receiver = Receiver(self.queue_receiver, num_sender=self.num, postprocess=self.post_process)
        while True:
            try:
                data = receiver.recv()
            except queue.Empty:
                logging.info("successfully receive all data!")
                self.stopped = True
                break

            while True:
                """
                只有成功put 数据，才修改send cnt
                """
                try:
                    self.output_queue.put(data, timeout=0.1)
                    break
                except queue.Full:
                    logging.debug("output_queue is full, the bottleneck is the speed of learner consume batch")







