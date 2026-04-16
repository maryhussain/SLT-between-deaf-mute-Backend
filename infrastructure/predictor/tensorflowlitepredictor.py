from application.interfaces.predictor_interface import PredictorInterface
import tensorflow as tf
import numpy as np
import joblib 


class TensorFlowLitePredictor(PredictorInterface):

    def __init__(self, model_path: str, scaler_path: str ,action_labels: list[str]):
        self.action_labels = action_labels
        #load the scaler
        self.scaler = joblib.load(scaler_path)
        #load the tflite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details() 
        
        

    def preprocess(self, sequence: list) -> np.ndarray:

        sequence = np.array(sequence, dtype=np.float32)
        if sequence.shape[0] != 40:
            raise ValueError("Invalid sequence length")
        
        scaled_sequence = self.scaler.transform(sequence)
        scaled_sequence= scaled_sequence.reshape(1,40, sequence.shape[-1])

        return scaled_sequence.astype(np.float32)

    def predict(self, data: np.ndarray) -> tuple[str, float]:

        self.interpreter.set_tensor(
             self.input_details[0]['index'],
               data.astype(np.float32)
             )
        
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(
             self.output_details[0]['index']
             )[0]
        
        label = self.action_labels[np.argmax(output_data)]
        confidence = float(np.max(output_data))
        
        return label, confidence