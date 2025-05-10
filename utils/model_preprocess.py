# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from PIL import Image
import os


def detect_face_and_landmarks(image_path):
    """Detect face using OpenCV cascade classifier"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use OpenCV face detector
    FACE_CASCADE_PATH = r'D:\college\MicroExpressionRecognizer\utils\haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    # Get first face
    x, y, w, h = faces[0]

    # Expand crop area
    padding = int(0.2 * max(w, h))
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)

    face_img = img[y_start:y_end, x_start:x_end]

    # Resize to 128x128
    face_img_resized = cv2.resize(face_img, (128, 128))

    # Convert to grayscale
    face_gray = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2GRAY)

    # Generate fake landmarks (since we're not using dlib)
    # Create 68 fake keypoints
    landmarks = np.random.rand(68, 2) * 128

    return face_gray, landmarks


def calculate_optical_flow(prev_gray, curr_gray):
    """Calculate optical flow"""
    try:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        return flow
    except:
        # Return zero flow if calculation fails
        return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)


def process_single_image(image_path):
    """Process single image and return model inputs"""
    # Detect face
    face_gray, landmarks = detect_face_and_landmarks(image_path)

    if face_gray is None:
        return None, None, None

    # Create fake image sequence (for single image case)
    image_sequence = np.array([face_gray for _ in range(32)])
    image_sequence = image_sequence[:, np.newaxis, :, :]  # Add channel dimension

    # Create fake landmark sequence
    landmarks_flat = landmarks.flatten()
    landmarks_sequence = np.array([landmarks_flat for _ in range(32)])

    # Create fake optical flow sequence (all zeros)
    flow_sequence = np.zeros((31, 2, 128, 128))

    return image_sequence, landmarks_sequence, flow_sequence


def process_video_for_model(video_path, max_frames=32):
    """Process video and extract image, landmark, and flow sequences"""
    cap = cv2.VideoCapture(video_path)

    frames = []
    landmarks_list = []
    gray_frames = []

    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Save temporary image
        temp_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(temp_path, frame)

        # Detect face and landmarks
        face_gray, landmarks = detect_face_and_landmarks(temp_path)

        # Delete temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        if face_gray is not None:
            frames.append(face_gray)
            landmarks_list.append(landmarks.flatten())
            gray_frames.append(face_gray)
            frame_count += 1

    cap.release()

    if len(frames) < 2:
        return None, None, None

    # Calculate optical flow
    flows = []
    for i in range(len(gray_frames) - 1):
        flow = calculate_optical_flow(gray_frames[i], gray_frames[i + 1])
        # Reshape flow to (2, H, W)
        flow = np.transpose(flow, (2, 0, 1))
        flows.append(flow)

    # Convert to numpy arrays
    image_sequence = np.array(frames)[:, np.newaxis, :, :]
    landmarks_sequence = np.array(landmarks_list)
    flow_sequence = np.array(flows)

    # Pad if not enough frames
    if len(frames) < max_frames:
        # Pad image sequence
        padding_size = max_frames - len(frames)
        image_padding = np.repeat(image_sequence[-1:], padding_size, axis=0)
        image_sequence = np.concatenate([image_sequence, image_padding], axis=0)

        # Pad landmark sequence
        landmarks_padding = np.repeat(landmarks_sequence[-1:], padding_size, axis=0)
        landmarks_sequence = np.concatenate([landmarks_sequence, landmarks_padding], axis=0)

        # Pad flow sequence
        flow_padding_size = max_frames - 1 - len(flows)
        if flow_padding_size > 0:
            flow_padding = np.zeros((flow_padding_size, 2, 128, 128))
            flow_sequence = np.concatenate([flow_sequence, flow_padding], axis=0)

    # Truncate to specified length
    image_sequence = image_sequence[:max_frames]
    landmarks_sequence = landmarks_sequence[:max_frames]
    flow_sequence = flow_sequence[:max_frames - 1]

    return image_sequence, landmarks_sequence, flow_sequence


def preprocess_webcam_frame(frame, prev_frame=None, prev_landmarks=None):
    """Process webcam frame"""
    # Save temporary image
    temp_path = "temp_webcam_frame.jpg"
    cv2.imwrite(temp_path, frame)

    # Detect face and landmarks
    face_gray, landmarks = detect_face_and_landmarks(temp_path)

    # Delete temporary file
    try:
        os.remove(temp_path)
    except:
        pass

    if face_gray is None:
        return None, None, None, None, None

    # Calculate optical flow (if previous frame exists)
    flow = None
    if prev_frame is not None:
        flow = calculate_optical_flow(prev_frame, face_gray)
        flow = np.transpose(flow, (2, 0, 1))

    return face_gray, landmarks.flatten(), flow, face_gray, landmarks