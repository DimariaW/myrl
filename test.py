from myrl.utils import set_process_logger
#from tests.test_actor import test_multiprocess_job_executor, test_multiprocess_batcher, Server
#from tests.test_logger import test_logger

from tests.test_actor.create_actor import open_gather, create_actor

if __name__ == "__main__":
    #test_multiprocess_job_executor()
    #test_logger()
    #test_multiprocess_batcher()
    #Server().run()
    set_process_logger()
    open_gather(host="127.0.0.1", port=8010, num_gathers=1, num_sample_actors_per_gather=4, func=create_actor)