import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import datetime

t = datetime.datetime.now()
# useful functions
def gidf(layer, cell):
    return wires[0:layer].sum() + cell

def eposf(layer, phi):
    for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['eastphi'][gid] - phi) < (3.143 / wires[layer]):
        return gid
    for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['eastphi'][gid] - phi) > (6.2832 - (3.143 / wires[layer])):
        return gid

def wposf(layer, phi):
    for gid in range(wiresum[layer - 1], wiresum[layer]):
        if abs(wiresangles['westphi'][gid] - phi) < (3.143 / wires[layer]):
            return gid
    for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['westphi'][gid] - phi) > (6.2832 - (3.143 / wires[layer])):
        return gid

# data
anglesnames=['eastphi','westphi', 'east_r', 'west_r']
anglesdtypes={'eastphi':np.float32,'westphi':np.float32,'east_r':np.float32,'west_r':np.float32}
wiresangles = pd.DataFrame()
wiresangles = pd.read_csv('./EndWireAngles.csv', header=0, sep=',', names=anglesnames, dtype=anglesdtypes)

# Wires at each layer
wires = np.array([40, 44, 48, 56, 64, 72, 80, 80, 76, 76, 88, 88, 100, 100, 112, 112, 128, 128, 140, 140, \
                 160, 160, 160, 160, 176, 176, 176, 176, 208, 208, 208, 208, 240, 240, 240, 240, \
                  256, 256, 256, 256, 288, 288, 288])
wiresum = np.cumsum(wires)
# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm
# 1st gap: 170mm to 189mm, 2nd gap: 385mm to 392mm, 3rd gap: about 652mm to 658mm
# radius of the last layer: about 760mm 


# array of the directly above neighbor of MDC wires at their east and west ends
eabove = np.empty(shape=(6508), dtype=int) 
wabove = np.empty(shape=(6508), dtype=int) 
for i in range(42):
    for j in range(wires[i]):
        gid = gidf(i, j)
        eabove[gid] = eposf(i + 1, wiresangles['eastphi'][gid])
        wabove[gid] = wposf(i + 1, wiresangles['westphi'][gid])

aboveposition = np.column_stack((np.arange(6508), eabove, wabove)).astype(np.int16)
np.savetxt("./results/aboveposition.txt", aboveposition, "%d, %d, %d", \
  header=("gid, above neighbor on the east end, above neighbor on the west end"))
np.savetxt("./results/aboveposition.csv", aboveposition, "%d, %d, %d", \
  header=("gid, above neighbor on the east end, above neighbor on the west end"))
print(aboveposition, aboveposition.shape)
print(datetime.datetime.now() - t)


fige = plt.figure(figsize=(350, 350))
ax = plt.subplot(211, projection='polar')
ax.set_yticks([7.5, 12.1, 17, 18.9, 38.5, 39.2, 45.2, 52.2, 58.2, 65.2, 65.8], \
    ['7.5 cm', '12.1 cm', '17.0 cm', '18.9 cm', '38.5 cm', '39.2 cm', '45.2 cm', '52.2 cm', '58.2 cm', '65.2 cm', '65.8 cm'], \
        position = (1.178, 0), color='b', fontsize=20)
trans_offset_p = mtransforms.offset_copy(ax.transData, fig=fige, y = 6, units='dots')
trans_offset_pe = mtransforms.offset_copy(ax.transData, fig=fige, y = 6, x = -13, units='dots')
trans_offset_pw = mtransforms.offset_copy(ax.transData, fig=fige, y = 6, x = 13, units='dots')
trans_offset_n = mtransforms.offset_copy(ax.transData, fig=fige, y = -20, units='dots')
plt.scatter(wiresangles['eastphi'], wiresangles['east_r'])
plt.title('\n\neast end view of the MDC wires\n the neighbor directly above each wire is indicated\n \
    (if it is different for the east and west ends, they are indicated with black and red respectively) \n\n', fontsize=75)
for gid in range(6796): 
    plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], gid, fontsize = 'x-small', \
        transform=trans_offset_n, horizontalalignment='center', verticalalignment='bottom')
for gid in range(6508): 
    if eabove[gid] == wabove[gid]:
        plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], eabove[gid], fontsize = 'x-small', \
            transform=trans_offset_p, horizontalalignment='center', verticalalignment='bottom')
    else:
        plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], eabove[gid], fontsize = 'x-small', \
            transform=trans_offset_pe, horizontalalignment='center', verticalalignment='bottom')
        plt.text(wiresangles['eastphi'][gid], wiresangles['east_r'][gid], wabove[gid], color='r', fontsize = 'x-small', \
            transform=trans_offset_pw, horizontalalignment='center', verticalalignment='bottom')
plt.savefig('./results/eabove.png', bbox_inches='tight')
print(datetime.datetime.now() - t)

figw = plt.figure(figsize=(350, 350))
ax = plt.subplot(211, projection='polar')
ax.set_yticks([7.5, 12.1, 17, 18.9, 38.5, 39.2, 45.2, 52.2, 58.2, 65.2, 65.8], \
    ['7.5 cm', '12.1 cm', '17.0 cm', '18.9 cm', '38.5 cm', '39.2 cm', '45.2 cm', '52.2 cm', '58.2 cm', '65.2 cm', '65.8 cm'], \
        position = (1.178, 0), color='b', fontsize=20)
#plt.setp(ax.yaxis.get_ticklabels(), rotation=-45 ) 
#ax.set_yticklabels(['7.5 cm', '12.1 cm', '17.0 cm', '18.9 cm', '38.5 cm', '39.2 cm', '45.2 cm', '52.2 cm', '58.2 cm', '65.2 cm', '65.8 cm'], position = (3*3.3/8, 0), color='b', fontsize=20) #, rotation = 45
#plt.setp(ax.yaxis.get_ticklabels(), rotation=-45 ) 
#plt.setp(ax.xaxis.get_ticklabels(), rotation=-45 ) 
#plt.setp(ax.get_yticklabels(), rotation=-45 ) 
#test1=ax.text(3.14/4, 7.5, '7.5 cm', rotation=-45, color='b', fontsize=40)
trans_offset_p = mtransforms.offset_copy(ax.transData, fig=figw, y = 6, units='dots')
trans_offset_pe = mtransforms.offset_copy(ax.transData, fig=figw, y = 6, x = -13, units='dots')
trans_offset_pw = mtransforms.offset_copy(ax.transData, fig=figw, y = 6, x = 13, units='dots')
trans_offset_n = mtransforms.offset_copy(ax.transData, fig=figw, y = -20, units='dots')
plt.scatter(wiresangles['westphi'], wiresangles['west_r'])
plt.title('\n\nwest end view of the MDC wires\n the neighbor directly above each wire is indicated\n \
    (if it is different for the east and west ends, they are indicated with red and black respectively) \n\n', fontsize=75)
for gid in range(6796):
    plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], gid, fontsize = 'x-small', \
        transform=trans_offset_n, horizontalalignment='center', verticalalignment='bottom')
for gid in range(6508):
    if eabove[gid] == wabove[gid]:
        plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], eabove[gid], fontsize = 'x-small', \
            transform=trans_offset_p, horizontalalignment='center', verticalalignment='bottom')
    else:
        plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], eabove[gid], color='r', fontsize = 'x-small', \
            transform=trans_offset_pe, horizontalalignment='center', verticalalignment='bottom')
        plt.text(wiresangles['westphi'][gid], wiresangles['west_r'][gid], wabove[gid], fontsize = 'x-small', \
            transform=trans_offset_pw, horizontalalignment='center', verticalalignment='bottom')
plt.savefig('./results/wabove.png', bbox_inches='tight')
print(datetime.datetime.now() - t)