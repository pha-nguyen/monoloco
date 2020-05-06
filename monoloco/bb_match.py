import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import time
from scipy.spatial.distance import cdist

def iou(bbox, candidates):

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def iou_cost(tracks, detections):

    track_indices = np.arange(len(tracks))
    detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        bbox = tracks[track_idx]
        candidates = np.asarray([detections[i] for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)

    return cost_matrix

def bb_match(tracker_bbs, bbs):
    tracker_bbs = np.array(tracker_bbs)
    bbs = np.array(bbs)
    track_indices = np.arange(len(tracker_bbs))
    detection_indices = np.arange(len(bbs))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return np.array([]).reshape((0, 2)), track_indices, detection_indices  # Nothing to match.
    # print(len(track_indices), ' X ', len(detection_indices))
    max_distance = 20
    levels = {}
    for t in track_indices:
        _level = 0
        if _level not in list(levels.keys()):
            levels[_level] = []
        levels[_level].append(t)

    level = max(list(levels.keys())) + 1
    matches, unmatched_tracks, unmatched_detections = [], [], []
    while len(matches) + len(unmatched_tracks) < len(tracker_bbs):
        level -= 1
        level = sorted([i for i in list(levels.keys()) if i <= level])[-1]
        _track_indices = levels[level].copy()

        if len(detection_indices) == 0:
            unmatched_tracks += _track_indices
            continue

        cost_matrix = cdist(tracker_bbs[_track_indices], bbs[detection_indices], 'euclidean')
        # complexity = max(complexity, len(_track_indices))
        used_cols = []
        for i, col in enumerate(cost_matrix.T):
            if np.any(col <= max_distance):
                used_cols.append(i)
            else:
                unmatched_detections.append(detection_indices[i])
        cost_matrix = cost_matrix[:, used_cols]
        detection_indices = detection_indices[used_cols]
        print('\t', level, cost_matrix.shape)
        indices = linear_assignment(cost_matrix)
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(_track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = _track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        detection_indices = np.array(unmatched_detections)
        # level = sorted([i for i in list(levels.keys()) if i < level])[-1]
        unmatched_detections = []
    # print(complexity)
    return np.array(matches), np.array(unmatched_tracks), np.array(detection_indices)