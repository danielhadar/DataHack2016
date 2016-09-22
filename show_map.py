#!/usr/bin/python3

import mplleaflet
import matplotlib.pyplot as plot
from config import nyc_bb

def bbox2vectors(bbox):
    coords = list(bbox.exterior.coords)
    return zip(*coords)

nyc_bb_x, nyc_bb_y = bbox2vectors(nyc_bb)
plot.plot(nyc_bb_x, nyc_bb_y)

mplleaflet.show()
