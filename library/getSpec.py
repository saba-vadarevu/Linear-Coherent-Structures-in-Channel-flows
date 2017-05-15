import numpy as np
import os
import miscUtil

os.environ['DATA'] = '/media/sabarish/channelData/R186/'

tArr = np.arange(124000,150050,50)
for t in tArr:
    miscUtil.phys2spec(t)
