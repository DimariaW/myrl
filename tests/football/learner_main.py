from tests.football.learner import MemoryReplayMain, LearnerMain, LeagueMain
from myrl.train import train_main
from tests.football.common import NAME

if __name__ == '__main__':
    mr_main = MemoryReplayMain(f"./log/{NAME}/")
    league_main = LeagueMain(f"./log/{NAME}/")
    leaner_main = LearnerMain(f"./log/{NAME}/")

    train_main(leaner_main, mr_main, league_main, queue_size=(8, 1))
