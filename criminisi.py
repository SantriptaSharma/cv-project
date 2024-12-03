import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import os

from utils import *

def select_parallel_lines_and_axis(img: np.ndarray):
	"""
	prompts the user to select 6 lines, pairwise parallel, outside-pair orthogonal to establish the (x, y, z) axes
	(x, y) must be on the reference plane and z must be out of the reference plane
	"""

	print("select 12 points, defining six lines x1, x2, y1, y2, z1, z2. x1[0] will be taken as the origin, with the axes (x1, y1, z1).\nEnsure that x1 || x2, y1 || y2, z1 || z2, and z1 is the direction we wish to make measurements in (out of the reference plane)")
	
	lines = select_lines(img, 6).reshape((3, 2, 2, 2))

	axes = lines[:, 0]
	origin = lines[0, 0, 0] 

	return origin, axes, lines

BASE_ANNOTATIONS_DIR = "criminisi_data"

def save_axes(name, origin, axes, lines):
	np.savez(os.path.join(BASE_ANNOTATIONS_DIR, name + "-annot.npz"), origin=origin, axes=axes, lines=lines)

def load_axes(name) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	annots = np.load(os.path.join(BASE_ANNOTATIONS_DIR, name + "-annot.npz"))

	return annots["origin"], annots["axes"], annots["lines"]

def save_measurements(name, lines, lengths):
	np.savez(os.path.join(BASE_ANNOTATIONS_DIR, name + "-lines.npz"), lines=lines, lengths=lengths)

def load_measurements(name) -> tuple[np.ndarray, np.ndarray]:
	meas = np.load(os.path.join(BASE_ANNOTATIONS_DIR, name + "-lines.npz"))

	return meas["lines"], np.array(meas["lengths"])

def draw_axes(img: np.ndarray, origin, axes) -> np.ndarray:
	axis_img = np.copy(img)

	lines = np.zeros((3, 2, 2))
	lines[:, 0] = origin

	for i, line in enumerate(axes):
		dir = line[1] - line[0]
		dir /= np.linalg.norm(dir)

		lines[i, 1] = origin + dir * 300
		lines[i, 1][lines[i, 1] < 0] = 0

	axis_img = draw_lines(axis_img, lines)

	return axis_img

def compute_vanishing_point(line_1, line_2):
	"""
	computes and returns the vanishing point of these (assumed parallel) lines in image space
	"""

	e1 = make_homogeneous(line_1[0])
	e2 = make_homogeneous(line_1[1])
	e3 = make_homogeneous(line_2[0])
	e4 = make_homogeneous(line_2[1])

	l1 = get_joining_line(e1, e2)
	l2 = get_joining_line(e3, e4)

	return intersect_lines(l1, l2)

def compute_vanishing_points(parallel_lines):
	vps = np.zeros((3, 3))

	for j in range(3):
		vps[:, j] = compute_vanishing_point(parallel_lines[j, 0], parallel_lines[j, 1])

	return vps

def get_proj_mat(vps):
	"""
	returns the projection matrix, upto an unknown scale
	"""

	P = np.zeros((3, 4))

	P[:, :3] = vps

	l_inf = get_joining_line(vps[:, 0], vps[:, 1])
	l_inf /= np.linalg.norm(l_inf)

	P[:, 3] = l_inf

	return P

def establish_scale(vertical_line, P, known_height):
	b = make_homogeneous(vertical_line[0])
	t = make_homogeneous(vertical_line[1])
	
	v = P[:, 2]
	l_inf = P[:, 3]

	alpha = -np.linalg.norm(np.cross(b, t)) / (np.dot(l_inf, b) * np.linalg.norm(np.cross(v, t)) * known_height)

	return alpha

def compute_height(vertical_line, P, alpha):
	b = make_homogeneous(vertical_line[0])
	t = make_homogeneous(vertical_line[1])
	
	v = P[:, 2]
	l_inf = P[:, 3]

	height = -np.linalg.norm(np.cross(b, t)) / (np.dot(l_inf, b) * np.linalg.norm(np.cross(v, t)) * alpha)

	return height


# def draw_vanishing_points(img, lines, vps = None):
# 	"""
# 	assumes lines are in endpoint form and vps in heterogeneous coords
# 	"""
	
# 	if vps is None:
# 		vps = []

# 		for i in range(len(lines), step = 2):
# 			vp = compute_vanishing_points(lines[i], lines[i + 1])
# 			vps.append(make_heterogeneous(vp))

# 	H, W = img.shape[:-1]

# 	min_pt = exp