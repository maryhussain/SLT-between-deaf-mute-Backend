from src.application.interfaces.predictor_interface import PredictorInterface
import tensorflow as tf
import numpy as np
from src.utils.preprocessing import normalize_sequence ,resample_sequence
from config import Config

class TensorFlowLitePredictor(PredictorInterface):

    def __init__(self, model_path: str, action_labels: list[str]):
        self.action_labels = action_labels
        self.model = tf.keras.models.load_model(model_path)
        # try:
        #     
        #     self.interpreter = tf.lite.Interpreter(model_path=model_path)
        #     self.interpreter.allocate_tensors()
        # except RuntimeError as e:
        #     if "Flex" in str(e) or "Select TensorFlow op" in str(e):
        #         print("تم رصد عمليات Flex، جاري تحميل الـ Interpreter مع الدعم الكامل للويندوز...") 
        #         self.interpreter = tf.lite.Interpreter(
        #             model_path=model_path,
        #             experimental_delegates=[tf.lite.experimental.load_delegate(None)] # تمرير None على الويندوز يجبر البيئة على استخدام الـ DLLs القياسية لتنسورفلو المدمج
        #         )
        #         self.interpreter.allocate_tensors()
        #     else:
        #         raise e
        
        #load the scaler
        # self.scaler = joblib.load(scaler_path)
        #load the tflite model
        
        # self.interpreter = tf.lite.Interpreter(model_path=model_path)
        # self.interpreter.allocate_tensors()

        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details() 
        
    def preprocess(self, sequence) -> np.ndarray:
        lh_seq, rh_seq, pose_seq = [], [], []
        for frame in sequence:
            frame_arr = np.array(frame)
            lh_seq.append(frame_arr[0:63]) 
            rh_seq.append(frame_arr[63:126])
            pose_seq.append(frame_arr[126:])

        # 1. normalize sequence
        seq = normalize_sequence(lh_seq, rh_seq, pose_seq)

        if len(seq) == 0:
            raise ValueError("Empty sequence after normalization")

        # 2. resample to fixed length
        seq_resampled = resample_sequence(seq, target_len=Config.NUM_FRAMES)

        # 3. reshape for model
        seq_resampled = seq_resampled.reshape(1, Config.NUM_FRAMES, Config.NUM_FEATURES)

        return seq_resampled.astype(np.float32)       

    # def preprocess(self, sequence: list) -> np.ndarray:

    #     sequence = np.array(sequence, dtype=np.float32)
    #     if sequence.shape[0] != Config.NUM_FRAMES or sequence.shape[1] != Config.NUM_FEATURES:
    #         raise ValueError("Invalid sequence length")
        
    #     scaled_sequence = self.scaler.transform(sequence)
    #     scaled_sequence= scaled_sequence.reshape(1,Config.NUM_CLASSES, sequence.shape[-1])

    #     return scaled_sequence.astype(np.float32)

    def predict(self, data: np.ndarray) -> tuple[str, float]:

        # self.interpreter.set_tensor(
        #      self.input_details[0]['index'],
        #        data.astype(np.float32)
        #      )
        
        # self.interpreter.invoke()

        # output_data = self.interpreter.get_tensor(
        #      self.output_details[0]['index']
        #      )[0]
        output_data = self.model.predict(data, verbose=0)[0]

        label = self.action_labels[np.argmax(output_data)]
        confidence = float(np.max(output_data))
        
        return label, confidence