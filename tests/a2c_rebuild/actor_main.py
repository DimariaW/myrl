from tests.a2c_rebuild.actor import ActorCreate
from myrl.core import open_gather

if __name__ == "__main__":
    logger_file_dir = "./log/lunar_lander/gathers"
    actor_main = ActorCreate(num_steps=32, logger_file_dir=logger_file_dir)

    open_gather(16, ("192.168.43.157", 7777),
                ("192.168.43.157", 7779),
                num_actors=1, actor_roles=["sampler"]*14 + ["evaluator"]*2, func=actor_main,
                use_bz2=False, logger_file_dir=logger_file_dir)

