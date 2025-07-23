from abc import abstractmethod

class DeviceRunner:

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def runInference(self, prompt: str, num_inference_steps: int = 50):
        pass

    @abstractmethod
    def close_device(self):
        pass