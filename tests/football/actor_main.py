from tests.football.actor import ActorCreate
from myrl.core import open_gather

import tests.football.config as cfg


if __name__ == "__main__":
    memory_server_index = 1  # 2, -1
    logger_file_dir = f"./log/{cfg.NAME}/gathers_{memory_server_index}/"
    actor_main = ActorCreate(logger_file_dir, steps=32)

    if memory_server_index == 1:
        open_gather(cfg.MEMORY1_ADDRESS, cfg.LEAGUE_ADDRESS, num_gathers=12, num_actors=4, actor_roles="sampler",
                    func=actor_main, use_bz2=cfg.USE_BZ2, logger_file_dir=logger_file_dir)

    elif memory_server_index == 2:
        open_gather(cfg.MEMORY2_ADDRESS, cfg.LEAGUE_ADDRESS, num_gathers=12, num_actors=4, actor_roles="sampler",
                    func=actor_main, use_bz2=cfg.USE_BZ2, logger_file_dir=logger_file_dir)

    else:
        open_gather(cfg.MEMORY1_ADDRESS, cfg.LEAGUE_ADDRESS, num_gathers=2, num_actors=12, actor_roles="evaluator",
                    func=actor_main, use_bz2=cfg.USE_BZ2, logger_file_dir=logger_file_dir)
