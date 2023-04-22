from osgeo import gdal
import glob, os
from PIL import Image
import numpy as np

path = "diverse/15TANZANIAc.jpg"
im = np.array(Image.open(path))

path = path[:-4]
#xmin, ymin, xmax, ymax = path.split(',')[:4]
#xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)

M = im.shape[0]//15
N = im.shape[1]//15

#width = float(xmax) - float(xmin)
#height = float(ymax) - float(ymin)

tiles = []
for y in range(0,im.shape[1],N):
    for x in range(0,im.shape[0],M):
        #bwidth = ((x+M) / im.shape[0]) * width
        #nwidth = ((x+M+1) / im.shape[0]) * width
        #bheight = ((y+N) / im.shape[1]) * height
        #nheight = ((y+N+1) / im.shape[1]) * height

        #new_xmin = round(xmin + bwidth,2)
        #new_ymin = round(ymin + bheight,2)
        #new_xmax = round(xmin + nwidth,2)
        #new_ymax = round(ymin + nheight,2)

        tile = Image.fromarray(im[x:x+M,y:y+N])
        tile.save(f'diverse/TANZANIA/{x},{y}c.jpg')
