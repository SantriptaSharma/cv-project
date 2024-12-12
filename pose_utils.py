import torch
import numpy as np
from ultralytics import YOLO


BODY_PARTS = { "Nose": 0, "LEye": 1, "REye": 2, "LCheek": 3, "RCheek": 4, "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9, "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14, "LAnkle": 15, "RAnkle": 16, "Background": 17}

# coarse, overdrawn (but also don't go up till temple)
# HEIGHT_CONTRIBUTING_LINES = [["LAnkle", "LKnee"], ["LKnee", "LHip"], ["LHip", "LShoulder"], ["LShoulder", "LEye"]]
# HEIGHT_CONTRIBUTING_LINES = [["LAnkle", "LKnee"], ["LKnee", "LHip"], ["LHip", "LEye"]]
HEIGHT_CONTRIBUTING_LINES = [["LAnkle", "LHip"], ["LHip", "LEye"]]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = None

def get_model():
	global model
	model = YOLO("checkpoints/yolo11x-pose.pt").to(device)


def get_poses(image):
	global model
	return model(image)[0].cpu()

def get_height_lines_by_object(pose):
	kps = pose.keypoints.xy.data
	
	objects = []

	for j in range(kps.shape[0]):
		lines = []

		for [a, b] in HEIGHT_CONTRIBUTING_LINES:
			idx_a = BODY_PARTS[a]
			idx_b = BODY_PARTS[b]

			lines.append([kps[j, idx_a], kps[j, idx_b]])

		objects.append(np.array(lines))

	return np.array(objects).astype(np.uint64)