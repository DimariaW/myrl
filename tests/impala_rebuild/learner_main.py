
from tests.impala_rebuild.learner import MemoryMain, LearnerMain, LeagueMain
from myrl.core import train_main


if __name__ == '__main__':
    name = "lunar_lander"
    mr_main1 = MemoryMain(7777, f"./log/{name}/")
    mr_main2 = MemoryMain(7778, f"./log/{name}/")
    league_main = LeagueMain(7779, f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, [mr_main1, mr_main2], league_main, memory_buffer_length=4)
