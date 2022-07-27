from myrl.core import train_main

from tests.test_myrl.test_memory_replay import MemoryReplayMain, LeagueMain, LearnerMain

if __name__ == '__main__':
    name = ["test_traj_list", "test_traj_queue", "test_traj_queue_mp"][2]
    mr_main = MemoryReplayMain(f"./log/{name}/")
    league_main = LeagueMain(f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, mr_main, league_main, queue_size=(3, 1))
