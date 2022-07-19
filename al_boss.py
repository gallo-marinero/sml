import numpy as np
from boss.bo.bo_main import BOMain
from boss.pp.pp_main import PPMain
from importlib import import_module

def boss(boss_inp):
    bo=BOMain.from_file(boss_inp)
    res=bo.run()
    #pp = PPMain(res)
    #pp.run()
