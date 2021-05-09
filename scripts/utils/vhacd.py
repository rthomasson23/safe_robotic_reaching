import pybullet as p
import pybullet_data as pd
import os

p.connect(p.DIRECT)
name_in = "../meshes/mobilefinger.obj"
name_out = "../meshes/mobilefinger_vhacd.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000 )

