
from tests.impala_rebuild.learner import MemoryMain, LearnerMain, LeagueMain
from myrl.core import train_main


if __name__ == '__main__':
    name = "lunar_lander"
    mr_main = MemoryMain(f"./log/{name}/")
    league_main = LeagueMain(f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, [mr_main], league_main, memory_buffer_length=2)
