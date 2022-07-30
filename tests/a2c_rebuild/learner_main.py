
from tests.a2c_rebuild.learner import MemoryMain, LearnerMain, LeagueMain
from myrl.core import train_main


if __name__ == '__main__':
    name = "lunar_lander"
    mm_main1 = MemoryMain(7777, f"./log/{name}/mm1")
    #mm_main2 = MemoryMain(7778, f"./log/{name}/mm2")
    league_main = LeagueMain(7779, f"./log/{name}/")
    leaner_main = LearnerMain(f"./log/{name}/")

    train_main(leaner_main, [mm_main1], [league_main], memory_buffer_length=1)