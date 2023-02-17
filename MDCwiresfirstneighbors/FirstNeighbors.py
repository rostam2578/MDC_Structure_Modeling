# building edges for the detailed Realistic Graph
import numpy as np
import pandas as pd
import math
from numpy import cumsum
#import torch
#device = torch.device('cuda')
#from torch_geometric.data import Data
import datetime
t = datetime.datetime.now()

# position of the wires at the endcaps
# (In fact we have the position of the sense wires at the ends (instead of z=0) and we can directly use this info.)
anglesnames=['eastphi','westphi', 'east_r', 'west_r']
anglesdtypes={'eastphi':np.float32,'westphi':np.float32,'east_r':np.float32,'west_r':np.float32}
wiresangles = pd.DataFrame()
wiresangles = pd.read_csv('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/MDCwiresgeometry/EndWireAngles.csv', header=0, sep=',', names=anglesnames, dtype=anglesdtypes)

# Wires at each layer
wires = np.array([40, 44, 48, 56, 64, 72, 80, 80, 76, 76, 88, 88, 100, 100, 112, 112, 128, 128, 140, 140, \
                 160, 160, 160, 160, 176, 176, 176, 176, 208, 208, 208, 208, 240, 240, 240, 240, \
                  256, 256, 256, 256, 288, 288, 288])
wiresum = cumsum(wires)
# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm
# 1st gap: 170mm to 189mm, 2nd gap: 385mm to 392mm, 3rd gap: about 652mm to 658mm
# radius of the last layer: about 760mm 

############################################################################
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

def laycelf(gid):
    layer = np.argmax(wiresum > gid)
    if layer == 0:
        return (0, gid)
    else:
        return layer, gid - wiresum[layer - 1]

# cell above
def egidabovef(gid, theta):
  phi = wiresangles['eastphi'][gid] + theta
  if phi < 0:
    phi = phi + 6.2834
  layer = laycelf(gid)[0]
  return eposf(layer + 1, phi)

def wgidabovef(gid, theta):
  phi = wiresangles['westphi'][gid] + theta
  if phi < 0:
    phi = phi + 6.2834
  layer = laycelf(gid)[0]
  return wposf(layer + 1, phi)

############################################################################
# Here we find first neighbors of each wire at two immediate neighboring layers.
# We choose the angle between two neighbors of the same layer and define the neighborhood with it.

# first is the beside wires on the same layer. Tis is the same all along the length of wires.
nextneighbors = []
neighborchart = np.empty(shape=(6976, 2), dtype=object)
for layer in np.arange(43):
  for cell in np.arange(wires[layer]):
    nextneighbors.extend([(gidf(layer, cell), gidf(layer, (cell + 1) % wires[layer]))]) 
allneighbors = nextneighbors
print('\nnumber of connections beween wires of the same layer', len(nextneighbors))

# straight to straight (17)
nextneighbors = []
straightwires = 0
for layer in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 36, 37, 38, 39, 40, 41]:
  straightwires += wires[layer]
  for cell in np.arange(wires[layer]):
    gid = gidf(layer, cell)
    gidabove = egidabovef(gid, 0)
    gidmax = gidabove
    gidmin = gidabove
    nextneighbors.extend([(gid, gidabove)])
    if (wiresangles['eastphi'][gidabove] + (3.1415 / wires[layer + 1])) < (wiresangles['eastphi'][gid] + (3.1415 / wires[layer])):
      gidmax = gidmax + 1
      if gidmax == wiresum[layer + 1]:
        gidmax = wiresum[layer]
      nextneighbors.extend([(gid, gidmax)])

    if (wiresangles['eastphi'][gidabove] - (3.1415 / wires[layer + 1])) > (wiresangles['eastphi'][gid] - (3.1415 / wires[layer])):
      gidmin = gidmin - 1
      if gidmin == wiresum[layer] - 1:
        gidmin = wiresum[layer + 1] - 1
      nextneighbors.extend([(gid, gidmin)])
    neighborchart[gid][0] = gidmax
    neighborchart[gid][1] = gidmin
allneighbors.extend(nextneighbors)
print('\nnumber of cells of straight layers with next layer also a straight layer', straightwires)
print('average number of neighbors', len(nextneighbors)/straightwires)

# stereo to stereo (22 layers)
nextneighbors = []
stereostereowires = 0
layer0 = np.array([0, 1, 2, 3, 4, 5, 6, 20, 21, 23, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
for i in np.arange(22):
  layer = layer0[i]
  stereostereowires += wires[layer]
  for cell in np.arange(wires[layer]):
    gid = gidf(layer, cell)
    if egidabovef(gid, 0) == wgidabovef(gid, 0):
      gidabove = egidabovef(gid, 0)
      gidmax = gidabove
      gidmin = gidabove
      nextneighbors.extend([(gid, gidabove)])
      if (wiresangles['eastphi'][gidabove] + (3.1415 / wires[layer + 1])) < (wiresangles['eastphi'][gid] + (3.1415 / wires[layer])):
        gidmax = gidmax + 1
        if gidmax == wiresum[layer + 1]:
          gidmax = wiresum[layer]
        nextneighbors.extend([(gid, gidmax)])
      if (wiresangles['eastphi'][gidabove] - (3.1415 / wires[layer + 1])) > (wiresangles['eastphi'][gid] - (3.1415 / wires[layer])):
        gidmin = gidmin - 1
        if gidmin == wiresum[layer] - 1:
          gidmin = wiresum[layer + 1] - 1
        nextneighbors.extend([(gid, gidmin)])
      neighborchart[gid][0] = gidmax
      neighborchart[gid][1] = gidmin
    elif abs(egidabovef(gid, 0) - wgidabovef(gid, 0)) < (wires[layer + 1] / 2):
      if egidabovef(gid, 0) > wgidabovef(gid, 0):
        gidmax = egidabovef(gid, 0)
        gidmin = wgidabovef(gid, 0)
        if wiresangles['eastphi'][gidmax] + (3.143 / wires[layer + 1]) < \
          wiresangles['eastphi'][gid] + (3.143 / wires[layer]):
          gidmax = gidmax + 1
          if gidmax == wiresum[layer + 1]:
            gidmax = wiresum[layer]
        if wiresangles['westphi'][gidmin] - (3.143 / wires[layer + 1]) > \
          wiresangles['westphi'][gid] - (3.143 / wires[layer]):
          gidmin = gidmin - 1
          if gidmin == wiresum[layer] - 1:
            gidmin = wiresum[layer + 1] - 1
      else:
        gidmax = wgidabovef(gid, 0)
        gidmin = egidabovef(gid, 0)
        if wiresangles['westphi'][gidmax] + (3.143 / wires[layer + 1]) < \
          wiresangles['westphi'][gid] + (3.143 / wires[layer]):
          gidmax = gidmax + 1
          if gidmax == wiresum[layer + 1]:
            gidmax = wiresum[layer]
        if wiresangles['eastphi'][gidmin] - (3.143 / wires[layer + 1]) > \
          wiresangles['eastphi'][gid] - (3.143 / wires[layer]):
          gidmin = gidmin - 1  
          if gidmin == wiresum[layer] - 1:
            gidmin = wiresum[layer + 1] - 1
      neighborchart[gid][0] = gidmax
      neighborchart[gid][1] = gidmin 
      if gidmax > gidmin:
        for gidabove in np.arange(gidmin, gidmax + 1):
          nextneighbors.extend([(gid, gidabove)])
      else:
        for gidabove in np.arange(gidmin, wiresum[layer + 1]):
          nextneighbors.extend([(gid, gidabove)])
        for gidabove in np.arange(wiresum[layer], gidmax + 1):
          nextneighbors.extend([(gid, gidabove)])
    else:
      if egidabovef(gid, 0) < wgidabovef(gid, 0):
        gidmax = egidabovef(gid, 0)
        gidmin = wgidabovef(gid, 0)
        if wiresangles['eastphi'][gidmax] + (3.143 / wires[layer + 1]) < \
          wiresangles['eastphi'][gid] + (3.143 / wires[layer]):
          gidmax = gidmax + 1
          if gidmax == wiresum[layer + 1]:
            gidmax = wiresum[layer]
        if wiresangles['westphi'][gidmin] - (3.143 / wires[layer + 1]) > \
          wiresangles['westphi'][gid] - (3.143 / wires[layer]):
          gidmin = gidmin - 1
          if gidmin == wiresum[layer] - 1:
            gidmin = wiresum[layer + 1] - 1
      else:
        gidmax = wgidabovef(gid, 0)
        gidmin = egidabovef(gid, 0)
        if wiresangles['westphi'][gidmax] + (3.143 / wires[layer + 1]) < \
          wiresangles['westphi'][gid] + (3.143 / wires[layer]):
          gidmax = gidmax + 1
          if gidmax == wiresum[layer + 1]:
            gidmax = wiresum[layer]
        if wiresangles['eastphi'][gidmin] - (3.143 / wires[layer + 1]) > \
          wiresangles['eastphi'][gid] - (3.143 / wires[layer]):
          gidmin = gidmin - 1  
          if gidmin == wiresum[layer] - 1:
            gidmin = wiresum[layer + 1] - 1
      neighborchart[gid][0] = gidmax
      neighborchart[gid][1] = gidmin        
      for gidabove in np.arange(gidmin, wiresum[layer + 1]):
        nextneighbors.extend([(gid, gidabove)])
      for gidabove in np.arange(wiresum[layer], gidmax + 1):
        nextneighbors.extend([(gid, gidabove)])
allneighbors.extend(nextneighbors)
print('\nnumber of cells of stereo layers with next layer also a stereo layer', stereostereowires)
print('average number of neighbors', len(nextneighbors)/stereostereowires)

# gaps, stereo to axial or vice versa (3 layers)
nextneighbors = []
axialstereowires = 0
gaptheta = np.array([math.acos(170/189), math.acos(385/392), math.acos(652/658)])# the gap is estimated to be between 170cm and 189cm
layer2 = np.array([7, 19, 35])
for i in np.arange(3):
  layer = layer2[i]
  axialstereowires += wires[layer]
  for cell in np.arange(wires[layer]):
    gid = gidf(layer, cell)  
    gidmax = wgidabovef(gid, gaptheta[i])
    gidmin = egidabovef(gid, -gaptheta[i])
    if (wiresangles['westphi'][gidmax] + (3.141 / wires[layer + 1])) < (wiresangles['westphi'][gid] + (3.142 / wires[layer])):
        gidmax = gidmax + 1
        if gidmax == wiresum[layer + 1]:
          gidmax = wiresum[layer]
    if wiresangles['eastphi'][gidmin] - (3.143 / wires[layer + 1]) > \
        wiresangles['eastphi'][gid] - (3.143 / wires[layer]):
        gidmin = gidmin - 1
        if gidmin == wiresum[layer] - 1:
          gidmin = wiresum[layer + 1] - 1
    neighborchart[gid][0] = gidmax
    neighborchart[gid][1] = gidmin
    if gidmax > gidmin:
      for gidabove in np.arange(gidmin, gidmax + 1):      
        nextneighbors.extend([(gid, gidabove)])
    else:
      for gidabove in np.arange(gidmin, wiresum[layer + 1]):
        nextneighbors.extend([(gid, gidabove)])
      for gidabove in np.arange(wiresum[layer], gidmax):
        nextneighbors.extend([(gid, gidabove)])
allneighbors.extend(nextneighbors)
print('\nangular shift due to the gap', gaptheta * 180 / 3.143)
print('number of cells of layers with a gap after their layer', axialstereowires)
print('average number of neighbors', len(nextneighbors)/axialstereowires)
print('\ntotal number of connections', len(allneighbors))
print('time passed:', datetime.datetime.now() - t)

############################################################################
# all the edges
edge_index = allneighbors

# form a table of the neighbors
dfedge = pd.DataFrame(np.array(edge_index), columns=['n1', 'n2'])
edgetable = []
for i in range(6796):
  edgetable.append([i, np.array(pd.concat((dfedge[dfedge.n1 == i]['n2'], dfedge[dfedge.n2 == i]['n1'])).sort_values())])
print('\nnumber of edges:', len(edge_index[1]), '\nAverage edges per node:', len(edge_index[1]) / 6796 * 2)

np.save('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/MDCwiresfirstneighbors/share/neighborchart.npy', neighborchart)
np.savetxt('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/MDCwiresfirstneighbors/results/neighborstable.txt', allneighbors,  fmt='%s')
np.save('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/MDCwiresfirstneighbors/share/edge_index.npy', edge_index)
np.savetxt('/hpcfs/bes/mlgpu/hoseinkk/MLTracking/bhabha/GNN/MDCwiresfirstneighbors/results/edgetable.txt', edgetable, fmt='%s')
