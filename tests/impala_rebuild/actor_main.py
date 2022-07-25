from tests.impala_rebuild.actor import ActorCreate
from myrl.actor import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander"
    actor_main = ActorCreate(logger_file_dir, 32)

    open_gather(("localhost", 7777), ("localhost", 7778), num_gathers=1,
                num_sample_actors_per_gather=3, num_predict_actors_per_gather=0,
                func=actor_main, logger_file_dir=logger_file_dir)

