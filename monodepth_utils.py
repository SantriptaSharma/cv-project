import numpy as np

def backproject(K, depth, x, y):
	fx = K[0, 0]
	fy = K[1, 1]
	cx = K[0, 2]
	cy = K[1, 2]
	
	x_3d = (x - cx) * depth / fx
	y_3d = (y - cy) * depth / fy

	return np.array([x_3d, y_3d, depth])

def make_measurements(K, depth, lines):
	measurements = []

	for line in lines:
		pt_a = line[0]
		pt_b = line[1]

		depth_a = depth[pt_a[1], pt_a[0]]
		depth_b = depth[pt_b[1], pt_b[0]]

		pt3d_a = backproject(K, depth_a, pt_a[0], pt_a[1])
		pt3d_b = backproject(K, depth_b, pt_b[0], pt_b[1])

		dist = np.linalg.norm(pt3d_a - pt3d_b) * 100

		measurements.append(dist)

	return [round(x, 2) for x in measurements]
