import cv2
from src.application.interfaces.predictor_interface import PredictorInterface
import tensorflow as tf
import numpy as np
import mediapipe as mp
from src.utils.preprocessing import normalize_sequence ,resample_sequence
from config import Config

class TensorFlowPredictor(PredictorInterface):

    def __init__(self, model_path: str, action_labels: list[str]):
        self.action_labels = action_labels

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.holistic = mp.solutions.holistic.Holistic(
                        static_image_mode=True, 
                        min_detection_confidence=0.05)


    def extract_keypoints(self, image_bytes: bytes) -> np.ndarray:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return np.zeros(Config.NUM_FEATURES) 

            # *******************************************
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image = cv2.flip(image, 1)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)

            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

            return np.concatenate([lh, rh, pose])
        
        except Exception as e:
            print(f"Error processing image: {e}")
            return np.zeros(Config.NUM_FEATURES) 
     
    def preprocess(self, images_bytes: list[bytes]) -> np.ndarray:
        lh_seq, rh_seq, pose_seq = [], [], []


        for image in images_bytes:
                keypoints = self.extract_keypoints(image)
                lh_seq.append(keypoints[:63])  
                rh_seq.append(keypoints[63:126])  
                pose_seq.append(keypoints[126:])

        seq = normalize_sequence(lh_seq, rh_seq, pose_seq)

        if len(seq) == 0 :
            raise ValueError("No valid frames found in the sequence.")
        seq_resampled = resample_sequence(seq, target_len=Config.NUM_FRAMES)
        seq_resampled = seq_resampled.reshape(1, Config.NUM_FRAMES, Config.NUM_FEATURES)

        return seq_resampled.astype(np.float32)

    def predict(self, data: np.ndarray) -> tuple[str, float]:

        output_data = self.model.predict(data, verbose=0)[0]

        label = self.action_labels[np.argmax(output_data)]
        confidence = float(np.max(output_data))
        
        return label, confidence