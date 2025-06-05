import cv2
import dlib
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import configparser
import random
from pathlib import Path
from keras.utils import to_categorical
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CASME2Processor:
    def __init__(self, config_path=None):
        """Initialize the CASME2 dataset processor with configuration."""
        # Load configuration
        self.config = self.load_config(config_path or 'config.ini')

        # Initialize directories
        self.init_directories()

        # Initialize face detection tools
        self.predictor, self.detector, self.face_cascade = self.init_detectors()

    def load_config(self, config_path):
        """Load configuration from file or create default if not exists."""
        config = configparser.ConfigParser()

        if not os.path.exists(config_path):
            # Create default configuration
            config['PATHS'] = {
                'raw_data_dir': "data/CASME2-RAW",
                'processed_dir': "data/processed",
                'sequences_dir': "data/sequences",
                'optical_flow_dir': "data/optical_flow",
                'landmarks_dir': "data/landmarks",
                'coding_excel': "data/CASME2-coding-20140508.xlsx",
                'face_cascade': "utils/haarcascade_frontalface_default.xml",
                'shape_predictor': "utils/shape_predictor_68_face_landmarks.dat"
            }
            config['PARAMETERS'] = {
                'train_ratio': '0.8',
                'image_size': '128',
                'sequence_length': '32',
                'min_sequences_per_class': '400',
                'apex_offset': '5',  # Frames to include around apex
                'augmentation_factor': '2'  # Number of augmented samples per original
            }
            config['CLASSES'] = {
                'excluded_classes': '',
                'valid_classes': ''
            }

            with open(config_path, 'w') as f:
                config.write(f)
            logger.info(f"Created default configuration file: {config_path}")

        config.read(config_path)
        return config

    def init_directories(self):
        """Create necessary directories if they don't exist."""
        for key in ['processed_dir', 'sequences_dir', 'optical_flow_dir', 'landmarks_dir']:
            directory = self.config['PATHS'][key]
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

    def init_detectors(self):
        """Initialize face detectors and landmark predictors."""
        try:
            shape_predictor_path = self.config['PATHS']['shape_predictor']
            face_cascade_path = self.config['PATHS']['face_cascade']

            if not os.path.exists(shape_predictor_path):
                logger.error(f"Face landmark predictor file does not exist: {shape_predictor_path}")
                raise FileNotFoundError(f"Cannot find face landmark predictor file: {shape_predictor_path}")

            if not os.path.exists(face_cascade_path):
                logger.error(f"Haar cascade classifier file does not exist: {face_cascade_path}")
                raise FileNotFoundError(f"Cannot find Haar cascade classifier file: {face_cascade_path}")

            predictor = dlib.shape_predictor(shape_predictor_path)
            detector = dlib.get_frontal_face_detector()
            face_cascade = cv2.CascadeClassifier(face_cascade_path)

            logger.info("Successfully initialized face detectors and landmark predictors")
            return predictor, detector, face_cascade
        except Exception as e:
            logger.error(f"Failed to initialize detectors: {str(e)}")
            raise

    def analyze_class_distribution(self):
        """Analyze class distribution in the dataset and determine valid classes."""
        logger.info("Analyzing class distribution in the dataset...")

        excel_path = self.config['PATHS']['coding_excel']
        if not os.path.exists(excel_path):
            logger.error(f"Coding Excel file not found: {excel_path}")
            raise FileNotFoundError(f"Coding Excel file not found: {excel_path}")

        df = pd.read_excel(excel_path)

        # Count emotion classes
        emotion_counts = {}
        sequence_info = defaultdict(list)
        valid_sequences = 0
        invalid_sequences = 0

        for _, row in df.iterrows():
            emotion = str(row['Estimated Emotion']).strip().lower()
            subject = str(row['Subject']).zfill(2)
            filename = str(row['Filename']).strip()

            try:
                # Process frame indices
                onset_value = str(row['OnsetFrame']).strip()
                apex_value = str(row['ApexFrame']).strip()
                offset_value = str(row['OffsetFrame']).strip()

                # Check for invalid values
                if any(val in ['/', '', 'NaN', 'nan', None] for val in [onset_value, apex_value, offset_value]):
                    logger.warning(f"Invalid sequence: Subject {subject}, Filename {filename}")
                    invalid_sequences += 1
                    continue

                # Convert to integers
                onset = int(float(onset_value))
                apex = int(float(apex_value))
                offset = int(float(offset_value))

                # Validate frame indices
                if onset < 0 or apex < 0 or offset < 0:
                    logger.warning(f"Negative frame indices: Subject {subject}, Filename {filename}")
                    invalid_sequences += 1
                    continue

                if not (onset <= apex <= offset):
                    logger.warning(f"Invalid frame order: Subject {subject}, Filename {filename}")
                    invalid_sequences += 1
                    continue

            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing frame values: Subject {subject}, Filename {filename}, Error: {str(e)}")
                invalid_sequences += 1
                continue

            # Count valid sequences by emotion
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1

            # Store sequence information
            sequence_info[emotion].append({
                'subject': subject,
                'filename': filename,
                'onset': onset,
                'apex': apex,
                'offset': offset
            })
            valid_sequences += 1

        # Sort emotions by frequency
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1])

        logger.info("Dataset analysis results:")
        logger.info(f"Valid sequences: {valid_sequences}")
        logger.info(f"Invalid sequences: {invalid_sequences}")
        logger.info(f"Total sequences: {valid_sequences + invalid_sequences}")

        logger.info("\nClass distribution:")
        for emotion, count in sorted_emotions:
            logger.info(f"{emotion}: {count} sequences")

        # Select valid classes (excluding the two least frequent classes as in the paper)
        if len(sorted_emotions) < 3:
            raise ValueError(f"Not enough emotion classes in dataset (found {len(sorted_emotions)})")

        excluded_classes = [emotion for emotion, _ in sorted_emotions[:2]]
        valid_classes = [emotion for emotion, _ in sorted_emotions[2:]]

        logger.info(f"\nExcluded classes: {excluded_classes}")
        logger.info(f"Valid classes: {valid_classes}")

        # Update configuration
        self.config['CLASSES']['excluded_classes'] = ','.join(excluded_classes)
        self.config['CLASSES']['valid_classes'] = ','.join(valid_classes)

        with open('config.ini', 'w') as f:
            self.config.write(f)

        return valid_classes, sequence_info

    def extract_face_region(self, img_path):
        """Extract and align the face region from an image."""
        try:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Could not read image: {img_path}")
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect face using dlib
            faces = self.detector(gray)
            if len(faces) == 0:
                # Fallback to OpenCV face detector
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) == 0:
                    logger.warning(f"No face detected in: {img_path}")
                    return None

                # Convert OpenCV rectangle to dlib rectangle
                x, y, w, h = faces[0]
                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                face = dlib_rect
            else:
                face = faces[0]

            # Get face landmarks
            landmarks = self.predictor(gray, face)

            # Calculate eyes position for alignment
            left_eye = (landmarks.part(36).x + landmarks.part(39).x) // 2, (
                        landmarks.part(36).y + landmarks.part(39).y) // 2
            right_eye = (landmarks.part(42).x + landmarks.part(45).x) // 2, (
                        landmarks.part(42).y + landmarks.part(45).y) // 2

            # Calculate eye distance to determine face size
            eye_distance = np.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)

            # Face width and height based on eye distance (following facial proportions)
            face_width = int(2.0 * eye_distance)
            face_height = int(2.2 * eye_distance)

            # Calculate face center
            face_center_x = (left_eye[0] + right_eye[0]) // 2
            face_center_y = (left_eye[1] + right_eye[1]) // 2 + int(0.2 * eye_distance)  # Shift down slightly

            # Calculate face crop coordinates
            x1 = max(0, face_center_x - face_width // 2)
            y1 = max(0, face_center_y - face_height // 2)
            x2 = min(image.shape[1], x1 + face_width)
            y2 = min(image.shape[0], y1 + face_height)

            # Crop the face
            face_img = image[y1:y2, x1:x2]

            # Resize to standard size
            image_size = int(self.config['PARAMETERS']['image_size'])
            face_img = cv2.resize(face_img, (image_size, image_size))

            return face_img

        except Exception as e:
            logger.error(f"Error extracting face from {img_path}: {e}")
            return None

    def extract_optical_flow(self, prev_img, next_img):
        """Extract optical flow between two grayscale images."""
        try:
            if prev_img is None or next_img is None:
                return None

            # Ensure images are grayscale
            if len(prev_img.shape) == 3:
                prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
            if len(next_img.shape) == 3:
                next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            return flow

        except Exception as e:
            logger.error(f"Error calculating optical flow: {e}")
            return None

    def preprocess_image(self, img):
        """Preprocess image for neural network input (grayscale, normalize)."""
        if img is None:
            return None

        # Convert to grayscale if color
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def apply_augmentation(self, image):
        """Apply data augmentation to a single image."""
        if image is None:
            return None

        # Make a copy to avoid modifying the original
        img = image.copy()

        # Random horizontal flip (50% chance)
        if random.random() > 0.5:
            img = cv2.flip(img, 1)

        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 1)

        # Random contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        img = np.clip((img - 0.5) * contrast_factor + 0.5, 0, 1)

        return img

    def process_sequence(self, sequence_path, onset, apex, offset, use_apex_only=True):
        """Process a micro-expression sequence to extract frames and features."""
        # Get all frames in the sequence
        frames = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith('.jpg')])

        if use_apex_only:
            # For apex-only approach, extract apex frame and onset frame
            apex_offset = int(self.config['PARAMETERS']['apex_offset'])

            # Find apex and onset frames
            apex_idx = min(apex, len(frames) - 1)
            onset_idx = max(0, min(onset, len(frames) - 1))

            # Extract only these two frames
            frame_indices = [onset_idx, apex_idx]

            # Load and process frames
            processed_frames = []
            for idx in frame_indices:
                frame_path = os.path.join(sequence_path, frames[idx])
                face_img = self.extract_face_region(frame_path)
                if face_img is not None:
                    processed_frame = self.preprocess_image(face_img)
                    if processed_frame is not None:
                        processed_frames.append(processed_frame)

            if len(processed_frames) != 2:
                return None, None, None

            # Calculate optical flow between onset and apex
            onset_frame, apex_frame = processed_frames
            flow = self.extract_optical_flow(onset_frame, apex_frame)

            # Stack frames and flow for return
            frames_data = np.stack([onset_frame, apex_frame])

            # Convert flow to magnitude and angle for better representation
            flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_angle = np.arctan2(flow[..., 1], flow[..., 0])

            # Scale and normalize flow features
            flow_magnitude = flow_magnitude / (np.max(flow_magnitude) + 1e-8)
            flow_angle = (flow_angle + np.pi) / (2 * np.pi)

            # Stack flow features
            flow_data = np.stack([flow_magnitude, flow_angle], axis=-1)

            return frames_data, flow_data, [onset_idx, apex_idx]

        else:
            # For full sequence processing
            sequence_length = int(self.config['PARAMETERS']['sequence_length'])

            # Select frames from onset to offset, with apex in the middle if possible
            if offset - onset + 1 >= sequence_length:
                # Sequence is long enough, center around apex
                mid_point = (sequence_length - 1) // 2
                apex_relative = apex - onset
                start_idx = max(0, min(onset + apex_relative - mid_point, len(frames) - sequence_length))
            else:
                # Sequence is too short, center it and pad if necessary
                start_idx = max(0, onset - (sequence_length - (offset - onset + 1)) // 2)

            end_idx = min(len(frames), start_idx + sequence_length)

            # If we couldn't get enough frames, return None
            if end_idx - start_idx < sequence_length:
                return None, None, None

            # Extract and process the selected frames
            processed_frames = []
            frame_indices = list(range(start_idx, end_idx))

            for idx in frame_indices:
                frame_path = os.path.join(sequence_path, frames[idx])
                face_img = self.extract_face_region(frame_path)
                if face_img is not None:
                    processed_frame = self.preprocess_image(face_img)
                    if processed_frame is not None:
                        processed_frames.append(processed_frame)

            # If any frame processing failed, return None
            if len(processed_frames) != sequence_length:
                return None, None, None

            # Calculate optical flow between consecutive frames
            flow_frames = []
            for i in range(1, len(processed_frames)):
                flow = self.extract_optical_flow(processed_frames[i - 1], processed_frames[i])
                if flow is not None:
                    # Convert flow to magnitude and angle
                    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    flow_angle = np.arctan2(flow[..., 1], flow[..., 0])

                    # Scale and normalize
                    flow_magnitude = flow_magnitude / (np.max(flow_magnitude) + 1e-8)
                    flow_angle = (flow_angle + np.pi) / (2 * np.pi)

                    # Stack as channels
                    flow_frame = np.stack([flow_magnitude, flow_angle], axis=-1)
                    flow_frames.append(flow_frame)
                else:
                    flow_frames.append(np.zeros((processed_frames[0].shape[0], processed_frames[0].shape[1], 2)))

            # Pad the first frame with zeros since there's no previous frame
            flow_frames.insert(0, np.zeros((processed_frames[0].shape[0], processed_frames[0].shape[1], 2)))

            # Stack frames and flow
            frames_data = np.stack(processed_frames)
            flow_data = np.stack(flow_frames)

            return frames_data, flow_data, frame_indices

    def create_sequence_samples(self, sequence_info, valid_classes, use_apex_only=True):
        """Create sequence samples from the dataset."""
        logger.info("Creating sequence samples...")

        raw_data_dir = self.config['PATHS']['raw_data_dir']
        min_sequences = int(self.config['PARAMETERS']['min_sequences_per_class'])

        sequence_samples = defaultdict(list)

        for emotion in valid_classes:
            sequences = sequence_info[emotion]
            num_sequences = len(sequences)

            # Calculate how many loops we need for sufficient samples
            loops_needed = max(1, int(np.ceil(min_sequences / num_sequences)))
            logger.info(f"{emotion}: {num_sequences} sequences, will loop {loops_needed} times")

            for loop in range(loops_needed):
                for i, seq in enumerate(sequences):
                    subject = seq['subject']
                    filename = seq['filename']
                    onset = seq['onset']
                    apex = seq['apex']
                    offset = seq['offset']

                    # Path to the sequence
                    sequence_path = os.path.join(raw_data_dir, f"sub{subject}", filename)
                    if not os.path.exists(sequence_path):
                        logger.warning(f"Sequence path not found: {sequence_path}")
                        continue

                    # Process the sequence
                    frames_data, flow_data, frame_indices = self.process_sequence(
                        sequence_path, onset, apex, offset, use_apex_only
                    )

                    if frames_data is None or flow_data is None:
                        continue

                    # Create a unique sequence ID
                    sequence_id = f"{subject}_{filename}_loop{loop}"

                    # Store the processed sequence
                    sequence_samples[emotion].append({
                        'id': sequence_id,
                        'frames': frames_data,
                        'optical_flow': flow_data,
                        'frame_indices': frame_indices,
                        'emotion': emotion,
                        'subject': subject,
                        'filename': filename,
                        'onset': onset,
                        'apex': apex,
                        'offset': offset
                    })

        # Print final sequence counts
        logger.info("Final sequence counts:")
        for emotion in valid_classes:
            count = len(sequence_samples[emotion])
            logger.info(f"{emotion}: {count} sequences")

        return sequence_samples

    def split_sequences(self, sequence_samples, valid_classes):
        """Split sequences into training and test sets."""
        logger.info("Splitting sequences into training and test sets...")

        train_ratio = float(self.config['PARAMETERS']['train_ratio'])

        train_sequences = []
        test_sequences = []

        for emotion in valid_classes:
            sequences = sequence_samples[emotion]

            # Get unique subjects
            subjects = list(set(seq['subject'] for seq in sequences))

            # Split subjects to ensure person-independent evaluation
            train_subjects, test_subjects = train_test_split(
                subjects, train_size=train_ratio, random_state=42
            )

            # Assign sequences based on subject
            for seq in sequences:
                if seq['subject'] in train_subjects:
                    train_sequences.append(seq)
                else:
                    test_sequences.append(seq)

        logger.info(f"Training set: {len(train_sequences)} sequences")
        logger.info(f"Test set: {len(test_sequences)} sequences")

        return train_sequences, test_sequences

    def save_sequences(self, train_sequences, test_sequences, valid_classes):
        """Save processed sequences to disk."""
        logger.info("Saving sequences...")

        sequences_dir = self.config['PATHS']['sequences_dir']
        augmentation_factor = int(self.config['PARAMETERS']['augmentation_factor'])

        # Create directories
        for dataset in ['train', 'test']:
            for emotion in valid_classes:
                dir_path = os.path.join(sequences_dir, dataset, emotion)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

        # Process and save training sequences with augmentation
        train_info = []
        for seq in tqdm(train_sequences, desc="Saving training sequences"):
            emotion = seq['emotion']
            seq_id = seq['id']
            frames = seq['frames']
            optical_flow = seq['optical_flow']

            # Save original sequence
            base_path = os.path.join(sequences_dir, 'train', emotion, seq_id)
            frames_path = f"{base_path}_frames.npy"
            flow_path = f"{base_path}_flow.npy"

            np.save(frames_path, frames)
            np.save(flow_path, optical_flow)

            # Record original sequence info
            emotion_idx = valid_classes.index(emotion)
            train_info.append({
                'seq_id': seq_id,
                'emotion': emotion,
                'emotion_idx': emotion_idx,
                'frames_path': frames_path,
                'flow_path': flow_path,
                'augmented': False
            })

            # Create augmented versions
            for aug_idx in range(augmentation_factor):
                aug_seq_id = f"{seq_id}_aug{aug_idx}"

                # Apply augmentation to frames
                aug_frames = np.array([self.apply_augmentation(frame) for frame in frames])

                # Apply similar augmentation to optical flow
                aug_flow = optical_flow.copy()
                if aug_idx % 2 == 0:  # Flip horizontal for even augmentations
                    aug_flow = aug_flow[:, :, ::-1, :]
                    aug_flow[..., 0] = -aug_flow[..., 0]  # Negate x component of flow

                # Save augmented sequence
                aug_base_path = os.path.join(sequences_dir, 'train', emotion, aug_seq_id)
                aug_frames_path = f"{aug_base_path}_frames.npy"
                aug_flow_path = f"{aug_base_path}_flow.npy"

                np.save(aug_frames_path, aug_frames)
                np.save(aug_flow_path, aug_flow)

                # Record augmented sequence info
                # Record augmented sequence info
                train_info.append({
                    'seq_id': aug_seq_id,
                    'emotion': emotion,
                    'emotion_idx': emotion_idx,
                    'frames_path': aug_frames_path,
                    'flow_path': aug_flow_path,
                    'augmented': True
                })

                # Process and save test sequences (no augmentation)
            test_info = []
            for seq in tqdm(test_sequences, desc="Saving test sequences"):
                emotion = seq['emotion']
                seq_id = seq['id']
                frames = seq['frames']
                optical_flow = seq['optical_flow']

                # Save sequence
                base_path = os.path.join(sequences_dir, 'test', emotion, seq_id)
                frames_path = f"{base_path}_frames.npy"
                flow_path = f"{base_path}_flow.npy"

                np.save(frames_path, frames)
                np.save(flow_path, optical_flow)

                # Record sequence info
                emotion_idx = valid_classes.index(emotion)
                test_info.append({
                    'seq_id': seq_id,
                    'emotion': emotion,
                    'emotion_idx': emotion_idx,
                    'frames_path': frames_path,
                    'flow_path': flow_path,
                })

            return train_info, test_info

        def generate_label_files(self, train_info, test_info, valid_classes):
            """Generate label files for training and testing."""
            logger.info("Generating label files...")

            # Save class mapping
            with open('class_mapping.txt', 'w') as f:
                for idx, emotion in enumerate(valid_classes):
                    f.write(f"{idx}: {emotion}\n")

            # Save training labels
            with open('train_labels.txt', 'w') as f:
                for info in train_info:
                    # Create one-hot encoding
                    one_hot = to_categorical(info['emotion_idx'], num_classes=len(valid_classes))
                    one_hot_str = ','.join(map(str, one_hot))

                    # Write: emotion_idx;frames_path;flow_path;one_hot_encoding
                    f.write(f"{info['emotion_idx']};{info['frames_path']};{info['flow_path']};{one_hot_str}\n")

            # Save test labels
            with open('test_labels.txt', 'w') as f:
                for info in test_info:
                    # Create one-hot encoding
                    one_hot = to_categorical(info['emotion_idx'], num_classes=len(valid_classes))
                    one_hot_str = ','.join(map(str, one_hot))

                    # Write: emotion_idx;frames_path;flow_path;one_hot_encoding
                    f.write(f"{info['emotion_idx']};{info['frames_path']};{info['flow_path']};{one_hot_str}\n")

            logger.info(f"Generated training label file: train_labels.txt ({len(train_info)} samples)")
            logger.info(f"Generated test label file: test_labels.txt ({len(test_info)} samples)")
            logger.info(f"Generated class mapping file: class_mapping.txt")

        def visualize_distribution(self, train_info, test_info, valid_classes):
            """Visualize class distribution in training and test sets."""
            logger.info("Visualizing data distribution...")

            # Count samples by emotion
            train_counts = {emotion: 0 for emotion in valid_classes}
            test_counts = {emotion: 0 for emotion in valid_classes}

            for info in train_info:
                train_counts[info['emotion']] += 1

            for info in test_info:
                test_counts[info['emotion']] += 1

            # Create visualization
            plt.figure(figsize=(15, 6))

            # Training set distribution
            plt.subplot(1, 2, 1)
            x = range(len(valid_classes))
            train_values = [train_counts[emotion] for emotion in valid_classes]
            bars = plt.bar(x, train_values)
            plt.xlabel('Emotion Classes')
            plt.ylabel('Number of Sequences')
            plt.title('Training Set Distribution')
            plt.xticks(x, valid_classes, rotation=45)

            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

            # Test set distribution
            plt.subplot(1, 2, 2)
            test_values = [test_counts[emotion] for emotion in valid_classes]
            bars = plt.bar(x, test_values)
            plt.xlabel('Emotion Classes')
            plt.ylabel('Number of Sequences')
            plt.title('Test Set Distribution')
            plt.xticks(x, valid_classes, rotation=45)

            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('class_distribution.png', dpi=300)
            plt.close()

            # Print statistics
            logger.info("\nFinal dataset statistics:")
            total_train = sum(train_values)
            total_test = sum(test_values)

            for i, emotion in enumerate(valid_classes):
                logger.info(f"{emotion}:")
                logger.info(f"  Training: {train_values[i]} sequences ({train_values[i] / total_train * 100:.1f}%)")
                logger.info(f"  Testing: {test_values[i]} sequences ({test_values[i] / total_test * 100:.1f}%)")
                logger.info(f"  Total: {train_values[i] + test_values[i]} sequences")

        def prepare_for_video_magnification(self, sequence_info, valid_classes):
            """Prepare sequences for video magnification (if applicable)."""
            logger.info("Preparing sequences for video magnification...")

            raw_data_dir = self.config['PATHS']['raw_data_dir']
            processed_dir = self.config['PATHS']['processed_dir']

            # Create directories for magnified videos
            for emotion in valid_classes:
                emotion_dir = os.path.join(processed_dir, 'magnified', emotion)
                if not os.path.exists(emotion_dir):
                    os.makedirs(emotion_dir)

            # Process each sequence
            prepared_sequences = []

            for emotion in valid_classes:
                sequences = sequence_info[emotion]

                for seq in tqdm(sequences, desc=f"Preparing {emotion} sequences"):
                    subject = seq['subject']
                    filename = seq['filename']
                    onset = seq['onset']
                    apex = seq['apex']
                    offset = seq['offset']

                    # Path to the sequence
                    sequence_path = os.path.join(raw_data_dir, f"sub{subject}", filename)
                    if not os.path.exists(sequence_path):
                        logger.warning(f"Sequence path not found: {sequence_path}")
                        continue

                    # Get all frames in the sequence
                    frames = sorted([f for f in os.path.join(sequence_path) if f.lower().endswith('.jpg')])

                    if len(frames) < 10:  # Minimum frames for magnification
                        logger.warning(f"Not enough frames in {sequence_path}")
                        continue

                    # Create a directory for the processed sequence
                    seq_dir = os.path.join(processed_dir, 'extracted', emotion, f"{subject}_{filename}")
                    if not os.path.exists(seq_dir):
                        os.makedirs(seq_dir)

                    # Extract frames from onset to offset
                    frame_paths = []
                    for i in range(max(0, onset - 5), min(len(frames), offset + 5)):
                        frame_path = os.path.join(sequence_path, frames[i])
                        face_img = self.extract_face_region(frame_path)

                        if face_img is not None:
                            # Save the extracted face
                            output_path = os.path.join(seq_dir, f"frame_{i:04d}.jpg")
                            cv2.imwrite(output_path, face_img)
                            frame_paths.append(output_path)

                    if len(frame_paths) > 0:
                        # Save sequence info
                        prepared_sequences.append({
                            'emotion': emotion,
                            'subject': subject,
                            'filename': filename,
                            'frames': frame_paths,
                            'onset': onset,
                            'apex': apex,
                            'offset': offset,
                            'output_dir': os.path.join(processed_dir, 'magnified', emotion, f"{subject}_{filename}")
                        })

            logger.info(f"Prepared {len(prepared_sequences)} sequences for magnification")
            return prepared_sequences

        def process_dataset(self, use_apex_only=True, apply_magnification=False):
            """Process the entire CASME2 dataset."""
            try:
                # Step 1: Analyze class distribution
                valid_classes, sequence_info = self.analyze_class_distribution()

                # Step 2: Prepare for video magnification if needed
                if apply_magnification:
                    prepared_sequences = self.prepare_for_video_magnification(sequence_info, valid_classes)
                    # Note: Video magnification would be applied externally
                    # This would generate magnified videos in the processed_dir/magnified directory

                # Step 3: Create sequence samples
                sequence_samples = self.create_sequence_samples(sequence_info, valid_classes, use_apex_only)

                # Step 4: Split into training and test sets
                train_sequences, test_sequences = self.split_sequences(sequence_samples, valid_classes)

                # Step 5: Save sequences
                train_info, test_info = self.save_sequences(train_sequences, test_sequences, valid_classes)

                # Step 6: Generate label files
                self.generate_label_files(train_info, test_info, valid_classes)

                # Step 7: Visualize data distribution
                self.visualize_distribution(train_info, test_info, valid_classes)

                logger.info("Dataset processing complete!")

                # Summary
                logger.info("\nProcessing Summary:")
                logger.info(f"Valid emotion classes: {len(valid_classes)}")
                logger.info(f"Training sequences: {len(train_info)}")
                logger.info(f"Test sequences: {len(test_info)}")
                logger.info(
                    f"Image size: {self.config['PARAMETERS']['image_size']}x{self.config['PARAMETERS']['image_size']}")
                logger.info("Data includes: Preprocessed image frames and optical flow features")

                return True

            except Exception as e:
                logger.error(f"Error processing dataset: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False

        # DataLoader class to load processed data for model training
        class CASME2DataLoader:
            def __init__(self, train_labels_file='train_labels.txt', test_labels_file='test_labels.txt',
                         batch_size=32, shuffle=True):
                """Initialize data loader with label files."""
                self.train_labels_file = train_labels_file
                self.test_labels_file = test_labels_file
                self.batch_size = batch_size
                self.shuffle = shuffle

                # Load labels
                self.train_data = self.load_labels(train_labels_file)
                self.test_data = self.load_labels(test_labels_file)

                # Initialize batch indices
                self.train_indices = np.arange(len(self.train_data))
                self.test_indices = np.arange(len(self.test_data))
                self.train_batch_idx = 0
                self.test_batch_idx = 0

                # Shuffle initially if requested
                if shuffle:
                    np.random.shuffle(self.train_indices)

            def load_labels(self, label_file):
                """Load labels and file paths from a label file."""
                data = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(';')
                        if len(parts) >= 4:
                            emotion_idx = int(parts[0])
                            frames_path = parts[1]
                            flow_path = parts[2]
                            one_hot = list(map(float, parts[3].split(',')))

                            data.append({
                                'emotion_idx': emotion_idx,
                                'frames_path': frames_path,
                                'flow_path': flow_path,
                                'one_hot': one_hot
                            })

                return data

            def reset(self):
                """Reset batch indices and shuffle training data."""
                self.train_batch_idx = 0
                self.test_batch_idx = 0
                if self.shuffle:
                    np.random.shuffle(self.train_indices)

            def get_num_classes(self):
                """Get the number of classes."""
                if len(self.train_data) > 0:
                    return len(self.train_data[0]['one_hot'])
                return 0

            def get_train_size(self):
                """Get the number of training samples."""
                return len(self.train_data)

            def get_test_size(self):
                """Get the number of test samples."""
                return len(self.test_data)

            def get_train_batch(self):
                """Get the next batch of training data."""
                if self.train_batch_idx >= len(self.train_data) // self.batch_size:
                    self.train_batch_idx = 0
                    if self.shuffle:
                        np.random.shuffle(self.train_indices)

                # Get batch indices
                start_idx = self.train_batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.train_data))
                batch_indices = self.train_indices[start_idx:end_idx]

                # Load batch data
                batch_frames = []
                batch_flow = []
                batch_labels = []

                for idx in batch_indices:
                    item = self.train_data[idx]

                    # Load frames
                    frames = np.load(item['frames_path'])
                    batch_frames.append(frames)

                    # Load optical flow
                    flow = np.load(item['flow_path'])
                    batch_flow.append(flow)

                    # Get label
                    batch_labels.append(item['one_hot'])

                self.train_batch_idx += 1

                return np.array(batch_frames), np.array(batch_flow), np.array(batch_labels)

            def get_test_batch(self):
                """Get the next batch of test data."""
                if self.test_batch_idx >= len(self.test_data) // self.batch_size:
                    self.test_batch_idx = 0

                # Get batch indices
                start_idx = self.test_batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.test_data))
                batch_indices = self.test_indices[start_idx:end_idx]

                # Load batch data
                batch_frames = []
                batch_flow = []
                batch_labels = []

                for idx in batch_indices:
                    item = self.test_data[idx]

                    # Load frames
                    frames = np.load(item['frames_path'])
                    batch_frames.append(frames)

                    # Load optical flow
                    flow = np.load(item['flow_path'])
                    batch_flow.append(flow)

                    # Get label
                    batch_labels.append(item['one_hot'])

                self.test_batch_idx += 1

                return np.array(batch_frames), np.array(batch_flow), np.array(batch_labels)

        # Main execution point
        if __name__ == "__main__":
            # Process the dataset
            processor = CASME2Processor()
            success = processor.process_dataset(
                use_apex_only=True,  # Use only apex frame and onset frame
                apply_magnification=False  # Set to True if you want to prepare for magnification
            )

            if success:
                logger.info("Dataset processing completed successfully!")
            else:
                logger.error("Failed to process dataset.")