from tests.impala_rebuild.actor import ActorCreate
from myrl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander/gathers"
    actor_main = ActorCreate(num_steps=48, logger_file_dir=logger_file_dir)

    open_gather(2, [("127.0.0.1", 7777), ("127.0.0.1", 7778)],
                ("127.0.0.1", 7779),
                num_actors=2, actor_roles="sampler", func=actor_main,
                use_bz2=False, logger_file_dir=logger_file_dir)

