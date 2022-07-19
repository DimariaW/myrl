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
    open_gather(host="192.168.43.157", port=58899,
                num_gathers=2, num_sample_actors_per_gather=16, num_predict_actors_per_gather=4,
                func=create_actor, logger_file_dir="./log/empty_goal/")
    #tests.connections.test_mp_job_executor.test_multiprocess_job_executor_v2()
