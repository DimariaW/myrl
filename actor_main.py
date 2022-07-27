from tests.football.actor import ActorCreate
from myrl.actor import open_gather

import tests.football.config as cm

if __name__ == "__main__":
    logger_file_dir = f"./log/{cm.NAME}/"
    actor_main = ActorCreate(logger_file_dir, steps=32)

    open_gather(("10.127.45.22", 7777), ("10.127.45.22", 7778), num_gathers=12,
                num_sample_actors_per_gather=4, num_predict_actors_per_gather=0,
                func=actor_main, use_bz2=True, logger_file_dir=logger_file_dir)
