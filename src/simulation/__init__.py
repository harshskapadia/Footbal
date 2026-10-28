"""Synthetic broadcast match with ground truth, for testing the pipeline without real footage.

The rendered figures are not realistic enough for a YOLO model, so the simulator
also writes "detections" (ground-truth boxes with noise, misses and false
positives) and pitch keypoints. Stages 3-8 run on those exactly as they would on
a real model's output, and their results are compared with the ground truth.
"""
