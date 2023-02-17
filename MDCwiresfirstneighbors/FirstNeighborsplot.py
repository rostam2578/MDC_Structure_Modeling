import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import datetime
t = datetime.datetime.now()


# position of the wires at the endcaps
# (In fact there is no need for calculations, \
#  we have the position of the sense wires at the ends (instead of z=0) and we can directly use this info.)
anglesnames=['eastphi','westphi', 'east_r', 'west_r']
anglesdtypes={'eastphi':np.float32,'westphi':np.float32,'east_r':np.float32,'west_r':np.float32}
wiresangles = pd.DataFrame()
wiresangles = pd.read_csv('../MDCwiresgeometry/EndWireAngles.csv', header=0, sep=',', names=anglesnames, dtype=anglesdtypes)
neighborchart = np.load('./share/neighborchart.npy', allow_pickle=True)

# Wires at each layer
wires = np.array([40, 44, 48, 56, 64, 72, 80, 80, 76, 76, 88, 88, 100, 100, 112, 112, 128, 128, 140, 140, \
                 160, 160, 160, 160, 176, 176, 176, 176, 208, 208, 208, 208, 240, 240, 240, 240, \
                  256, 256, 256, 256, 288, 288, 288])
wiresum = np.cumsum(wires)
# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm
# 1st gap: 170mm to 189mm, 2nd gap: 385mm to 392mm, 3rd gap: about 652mm to 658mm
# radius of the last layer: about 760mm 


fige = plt.figure(figsize=(350, 350))
ax = plt.subplot(211, projection='polar')
ax.set_yticks([7.5, 12.1, 17, 18.9, 38.5, 39.2, 45.2, 52.2, 58.2, 65.2, 65.8], \
    ['7.5 cm', '12.1 cm', '17.0 cm', '18.9 cm', '38.5 cm', '39.2 cm', '45.2 cm', '52.2 cm', '58.2 cm', '65.2 cm', '65.8 cm'], \
        position = (1.178, 0), color='b', fontsize=20)
trans_offset_p = mtransforms.offset_copy(ax.transData, fig=fige, y = 6, units='dots')
trans_offset_n = mtransforms.offset_copy(ax.transData, fig=fige, y = -20, units='dots')
plt.scatter(wiresangles['eastphi'], wiresangles['east_r'])
plt.title('\n\neast end of the mdc wires\n (maxgid_neighbor, mingid_neighbor)\n gid\n karimi@ihep.ac.cn\n\n', fontsize=75)
for gid in range(6796):
    plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], gid, fontsize = 'x-small', \
        transform=trans_offset_n, horizontalalignment='center', verticalalignment='bottom')
for gid in range(6508):
    plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], (neighborchart[gid][0], neighborchart[gid][1]), fontsize = 'x-small', \
        transform=trans_offset_p, horizontalalignment='center', verticalalignment='bottom')
plt.savefig('./results/neighborsoneastview.png', bbox_inches='tight')

figw = plt.figure(figsize=(350, 350))
ax = plt.subplot(211, projection='polar')
ax.set_yticks([7.5, 12.1, 17, 18.9, 38.5, 39.2, 45.2, 52.2, 58.2, 65.2, 65.8], \
    ['7.5 cm', '12.1 cm', '17.0 cm', '18.9 cm', '38.5 cm', '39.2 cm', '45.2 cm', '52.2 cm', '58.2 cm', '65.2 cm', '65.8 cm'], \
        position = (1.178, 0), color='b', fontsize=20)
trans_offset_p = mtransforms.offset_copy(ax.transData, fig=figw, y = 6, units='dots')
trans_offset_n = mtransforms.offset_copy(ax.transData, fig=figw, y = -20, units='dots')
plt.scatter(wiresangles['westphi'], wiresangles['west_r'])
plt.title('\n\nwest end of the mdc wires\n (maxgid_neighbor, mingid_neighbor)\n gid\n karimi@ihep.ac.cn\n\n', fontsize=75)
for gid in range(6796):
    plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], gid, fontsize = 'x-small', \
        transform=trans_offset_n, horizontalalignment='center', verticalalignment='bottom')
for gid in range(6508):
    plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], (neighborchart[gid][0], neighborchart[gid][1]), fontsize = 'x-small', \
        transform=trans_offset_p, horizontalalignment='center', verticalalignment='bottom')
plt.savefig('./results/neighborsonwestview.png', bbox_inches='tight')
