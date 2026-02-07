import numpy as np

file = "/scratch/cl927/nutritionverse-3d-new/_NEWCODE_test_id-11-red-apple/vggt_camera_poses.npz"
data = np.load(file)
extrinsic = data["extrinsic"]


print(extrinsic)
