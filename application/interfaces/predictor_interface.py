from abc import ABC, abstractmethod
import numpy as np

class PredictorInterface(ABC):

    @abstractmethod
    def preprocess(self, sequence: list) -> np.ndarray :
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> tuple[str, float]: 
        pass