#from myrl.utils import set_process_logger
#from tests.test_actor import test_multiprocess_job_executor, test_multiprocess_batcher, Server
#from tests.test_logger import test_logger
from tests.test_football.actor import open_gather, create_actor

#import tests.connections.test_mp_job_executor

if __name__ == "__main__":
    #test_multiprocess_job_executor()
    #test_logger()
    #test_multiprocess_batcher()
    #Server().run()
    #set_process_logger()
    open_gather(host="10.127.45.22", port=58899,
                num_gathers=10, num_sample_actors_per_gather=4, num_predict_actors_per_gather=0,
                func=create_actor, logger_file_dir="./log/11_vs_11_easy_stochastic/")
    #tests.connections.test_mp_job_executor.test_multiprocess_job_executor_v2()