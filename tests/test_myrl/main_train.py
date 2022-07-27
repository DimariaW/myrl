"""
there are some bug in pycharm, all test should start from this file
"""
from myrl.core import train_main

from tests.test_myrl.test_train import MemoryReplayMain, LeagueMain, LearnerMain

if __name__ == '__main__':
    mr_main = MemoryReplayMain("./log/test_train/")
    league_main = LeagueMain("./log/test_train/")
    leaner_main = LearnerMain("./log/test_train/")

    train_main(leaner_main, mr_main, league_main, queue_size=(8, 2))

