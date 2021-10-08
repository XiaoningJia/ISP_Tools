import numpy as np
import imaging
import utility
import os,sys

image_name = 'rawdump_1920x1080_330'

temp = np.fromfile("images/" + image_name + ".pgm", dtype="uint16", sep="")

print(temp.shape())