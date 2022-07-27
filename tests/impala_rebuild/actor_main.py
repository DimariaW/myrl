from tests.impala_rebuild.actor import ActorCreate
from myrl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander"
    actor_main = ActorCreate(logger_file_dir, steps=32)

    open_gather(("172.18.237.35", 7777), ("172.18.237.35", 7778), num_gathers=1,
                num_actors=2, actor_roles="sampler",
                func=actor_main,
                use_bz2=True,
                logger_file_dir=logger_file_dir)

