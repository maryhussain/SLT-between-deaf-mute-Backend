import tensorflow as tf
import numpy as np
import joblib 


class TensorFlowLitePredictor:

    def __init__(self, model_path: str, scaler_path: str ,action_labels: list[str]):
        self.action_labels = action_labels
        #load the scaler
        self.scaler = joblib.load(scaler_path)
        #load the tflite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details() 
        
        

    def preprocess(self, sequence: np.ndarray) -> np.ndarray:

            X_scaled = self.scaler.transform(sequence)
            X_scaled = X_scaled.reshape(1,40, sequence.shape[-1])

            return X_scaled.astype(np.float32)

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