import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# PROJECTIVE GEOMETRY UTILS
# conversions
make_homogeneous = lambda pt: np.r_[pt, 1]
make_heterogeneous = lambda pt: (pt / pt[-1])[:-1]

# for pts in homogeneous coords
get_joining_line = lambda pt1, pt2: np.cross(pt1, pt2)
intersect_lines = get_joining_line


# PLOTTING & VIZ UTILS
get_color = lambda i: (125 * (i % 4), 125 * ((i + 1) % 4), 125 * ((i + 2) % 4))

def imshow(img: np.ndarray, ax = None):
	if ax is None:
		ax = plt

	ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap="gray")

def select_lines(img: np.ndarray, n: int = 1):
	plt.switch_backend("qt5agg")

	imshow(img)
	pts = plt.ginput(n * 2, 1000000)

	# line x point x position
	lines = np.zeros((n, 2, 2))

	for i in range(n):
		lines[i][0] = np.array(pts[i * 2])
		lines[i][1] = np.array(pts[i * 2 + 1])

	plt.switch_backend("inline")

	return lines

def draw_lines(img: np.ndarray, lines: np.ndarray, lengths = None, units = "cm"):
	new_img = np.copy(img)

	for i, line in enumerate(lines):
		a = tuple(line[0].astype(np.uint32))
		b = tuple(line[1].astype(np.uint32))

		cv.circle(new_img, a, 30, get_color(i * 7), 40)
		cv.circle(new_img, b, 30, get_color(i * 7), 40)
		cv.line(new_img, a, b, get_color(i * 5), 20)

		if lengths is not None:
			mid = ((a[0] + b[0])//2, (a[1] + b[1])//2)
			length = lengths[i]

			if length is None:
				continue

			cv.putText(new_img, f"{length}{units}", (mid[0] + 55, np.int32(mid[1]) + np.random.randint(-200, 201)), cv.FONT_HERSHEY_SIMPLEX, 4, get_color(21), 8)

	return new_img
