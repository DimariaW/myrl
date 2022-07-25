
from tests.impala_rebuild.learner import MemoryReplayMain, LearnerMain, LeagueMain
from myrl.train import train_main


if __name__ == '__main__':
    name = "lunar_lander"
    mr_main = MemoryReplayMain(f"./log/{name}/")
    league_main = LeagueMain(f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, mr_main, league_main, queue_size=(2, 1))
