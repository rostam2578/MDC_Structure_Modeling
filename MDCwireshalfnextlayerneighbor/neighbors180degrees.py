# building edges for the detailed Realistic Graph
import numpy as np
import pandas as pd
import math
#from numpy import cumsum
#import torch
#device = torch.device('cuda')
#from torch_geometric.data import Data
import datetime
t = datetime.datetime.now()

#########################################################
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
wiresum = np.cumsum(wires)
wiresforth = np.int16(wires / 4)
# The diameter of the wires are about 12 mm. First layer is in 71mm radius of the center.
# 1st tube radius: about 70mm to 170mm
# 1st gap: 170mm to 189mm, 2nd gap: 385mm to 392mm, 3rd gap: about 652mm to 658mm
# radius of the last layer: about 760mm 

#########################################################
# useful functions
def gidf(layer, cell):
    return wires[0:layer].sum() + cell

def laycelf(gid):
    layer = np.argmax(wiresum > gid)
    if layer == 0:
        return (0, gid)
    else:
        return layer, gid - wiresum[layer - 1]

# finds gid of a cell using layer and angle
def eposf(layer, phi):
    for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['eastphi'][gid] - phi) < (3.143 / wires[layer]):
        return int(gid)
    for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['eastphi'][gid] - phi) > (6.2832 - (3.143 / wires[layer])):
        return int(gid)

# cell above
def egidabovef(gid, theta):
  phi = wiresangles['eastphi'][gid] + theta
  if phi < 0:
    phi = phi + 6.2834
  layer = laycelf(gid)[0]
  return eposf(layer + 1, phi)

def wposf(layer, phi):
  for gid in range(wiresum[layer - 1], wiresum[layer]):
      if abs(wiresangles['westphi'][gid] - phi) < (3.143 / wires[layer]):
          return gid
  for gid in range(wiresum[layer - 1], wiresum[layer]):
    if abs(wiresangles['westphi'][gid] - phi) > (6.2832 - (3.143 / wires[layer])):
      return gid

def wgidabovef(gid, theta):
  phi = wiresangles['westphi'][gid] + theta
  if phi < 0:
    phi = phi + 6.2834
  layer = laycelf(gid)[0]
  return wposf(layer + 1, phi)

# half of the cells of any layer around a cell of the same layer.
# since gid range can be discontinuous, this function returns to range. 
def thesamelayer180d(gid):
  layer, cell = laycelf(gid)
  wiresum1 = np.zeros(44)
  wiresum1[0:43] = np.copy(wiresum)
  if cell < wires[layer] / 4:
    gidmin1 = wiresum1[layer - 1]
    gidmax1 = wiresum1[layer - 1] + wiresforth[layer] + cell
    gidmin2 = wiresum1[layer] - wiresforth[layer] + cell
    gidmax2 = wiresum1[layer] - 1
  elif cell > (3 * wires[layer] / 4 - 1):
    gidmin1 = wiresum1[layer - 1]
    gidmax1 = wiresum1[layer - 1] - 3 * wiresforth[layer] + cell
    gidmin2 = wiresum1[layer - 1] - wiresforth[layer] + cell
    gidmax2 = wiresum1[layer] - 1
  else:
    gidmin1 = wiresum1[layer -1] + cell - wires[layer] / 4
    gidmax1 = wiresum1[layer -1] + cell + wires[layer] / 4
    gidmin2 = 0
    gidmax2 = 0
  return int(gidmin1), int(gidmax1), int(gidmin2), int(gidmax2)

#########################################################
# Here we find first neighbors of each wire at two immediate neighboring layers.
# We choose the angle between two neighbors of the same layer and define the neighborhood with it.

# First is the beside wires on the same layer. This is the same all along the length of wires.
# we choose two imediate neighbors of at each side (to not repeat cunting, only right side is calculated).
nextneighbors = []
for layer in np.arange(43):
  for cell in np.arange(wires[layer]):
    nextneighbors.extend([(gidf(layer, cell), gidf(layer, (cell + 1) % wires[layer]))]) 
    nextneighbors.extend([(gidf(layer, cell), gidf(layer, (cell + 2) % wires[layer]))]) 
allneighbors = nextneighbors
print('\nnumber of connections beween wires of the same layer', len(nextneighbors))

# Prepare a chart indicating minimum and maximum of the neighbors on the next layer for each gid.
neighborchart = np.zeros(shape=(6508, 5), dtype=np.int32)
for gid in range(6508):
  abovegid = egidabovef(gid, 0)
  neighborchart[gid][0] = gid
  neighborchart[gid][1:5] = thesamelayer180d(abovegid)

# Next layer(s) neighbors.
# We choose east end view of the wires. Since we are counting half of the 
# cells of the next layer(s) as nighbors, we get more than real number of neighbors.
nextneighbors = []
ifurther = 1  # we can also do this for more layers ahead
for layer in np.arange(0, 42):
  for cell in np.arange(wires[layer]):
    gid = gidf(layer, cell)
    abovegid = gid
    for i in np.arange(layer, min(layer + ifurther, 42)):
      abovegid = egidabovef(abovegid, 0) 
      gidmin1, gidmax1, gidmin2, gidmax2 = thesamelayer180d(abovegid)
      for igid in np.arange(gidmin1, gidmax1 + 1):
        nextneighbors.extend([(gid, igid)])
      if gidmin2 != 0:
        for igid in np.arange(gidmin2, gidmax2 + 1):
          nextneighbors.extend([(gid, igid)])
allneighbors.extend(nextneighbors)
print('\ntotal number of connections to next layer(s)', len(nextneighbors))
print('\ntotal number of connections', len(allneighbors))
print('time passed:', datetime.datetime.now() - t)

#########################################################
# all the edges
edge_index = allneighbors

# form a table of the neighbors
dfedge = pd.DataFrame(np.array(edge_index), columns=['n1', 'n2'])
edgetable = []
for i in range(6796):
  edgetable.append([i, np.array(pd.concat((dfedge[dfedge.n1 == i]['n2'], dfedge[dfedge.n2 == i]['n1'])).sort_values())])
print('\nnumber of edges:', len(edge_index[1]), '\nAverage edges per node:', len(edge_index[1]) / 6796 * 2)

np.save('./share/neighborchart180d.npy', neighborchart)
np.savetxt('./results/neighborchart180d.txt', neighborchart, fmt='%s')
np.savetxt('./results/neighborstable180d.txt', allneighbors, fmt='%s')
np.save('./share/neighborstable180d.npy', allneighbors)
np.savetxt('./results/edgetable180d.txt', edgetable, fmt='%s')
np.save('./share/edge_index180d.npy', allneighbors)
