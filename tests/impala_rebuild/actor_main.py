from tests.impala_rebuild.actor import ActorCreate
from myrl.actor import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander"
    actor_main = ActorCreate(logger_file_dir, steps=256)

    open_gather(("172.18.237.51", 7777), ("172.18.237.51", 7778), num_gathers=1,
                num_sample_actors_per_gather=4, num_predict_actors_per_gather=0,
                func=actor_main, logger_file_dir=logger_file_dir)

