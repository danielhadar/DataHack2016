#!/usr/bin/python

import sys
import pandas
from shapely.geometry import Point, box

if len(sys.argv) != 3:
    print("usage: " + sys.argv[0] + " <src_csv> <dest_csv>")
    exit(1)


# From https://github.com/mapzen/metroextractor-cities/blob/master/cities.json
nyc_bb = box(-74.501, 40.345, -73.226, 41.097)

def in_nyc(row):
    from_point = Point(row['from_longitude'], row['from_latitude'])
    to_point = Point(row['to_longitude'], row['to_latitude'])
    return nyc_bb.contains(from_point) & nyc_bb.contains(to_point)

df = pandas.read_csv(sys.argv[1])
good_df = df[df.apply(in_nyc, axis = 1)].copy()
good_df.to_csv(sys.argv[2])
