from myrl.learner_server import ActorCommunicator
from myrl.utils import set_process_logger
set_process_logger()

actor_communicator = ActorCommunicator(1234)

actor_communicator.run()

while True:
    conn, msg = actor_communicator.recv()
    print(msg)

#%%

from myrl.connection import connect_socket_connection

conn = connect_socket_connection("172.18.237.67", 1234)
