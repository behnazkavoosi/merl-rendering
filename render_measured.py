import mitsuba as mi
import drjit as dr
import numpy as np
import gc
from measured_brdf_tensor import MyBSDF
from tqdm import tqdm
import torch
import os
import sys
import random 

mi.set_variant('cuda_ad_rgb')  
mi.set_log_level(mi.LogLevel.Error)
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)
mi.register_bsdf("mybsdf", lambda props: MyBSDF(props))

material = sys.argv[1]
scene_file = sys.argv[2]
scene_directory = f"D:/Projects/Paper05/brdf-rendering/mitsuba/render/scenes/measured/ref/{material}"
output_directory = f"D:/Projects/Paper05/brdf-rendering/mitsuba/render/renders/measured/ref/{material}/rgb"

os.makedirs(output_directory, exist_ok=True)

scene_path = os.path.join(scene_directory, scene_file)
print(f"Rendering {scene_file}...")

scene = mi.load_file(scene_path)
sensors = scene.sensors()
#left_camera = next((s for s in sensors if s.id() == 'left_camera'), None)
right_camera = next((s for s in sensors if s.id() == 'right_camera'), None)

SPP = 8
spp = SPP * 128

#left_image_sum = np.zeros_like(mi.render(scene, spp=SPP, sensor=left_camera))
right_image_sum = np.zeros_like(mi.render(scene, spp=SPP, sensor=right_camera))

#seed = 0
for _ in tqdm(range(spp // SPP)):
    seed = random.randint(0, 2**32 - 1)

    #left_image_part = mi.render(scene, spp=SPP, seed=seed, sensor=left_camera)
    right_image_part = mi.render(scene, spp=SPP, seed=seed, sensor=right_camera)

    #left_image_sum += left_image_part
    right_image_sum += right_image_part
    #seed += 1

    #del left_image_part, right_image_part 
    del right_image_part
    torch.cuda.empty_cache()  
    gc.collect()

#left_image = left_image_sum / (spp // SPP)
right_image = right_image_sum / (spp // SPP)

#left_output_path = os.path.join(output_directory, f"{scene_file[:-4]}_left_final.exr")
right_output_path = os.path.join(output_directory, f"{scene_file[:-4]}_right_final.exr")
#mi.util.write_bitmap(left_output_path, left_image)
mi.util.write_bitmap(right_output_path, right_image)


#del left_image_sum, right_image_sum, left_image, right_image, scene, sensors
del right_image_sum, right_image, scene, sensors
torch.cuda.empty_cache()  
gc.collect()

print(f"Finished rendering {scene_file}")



