import numpy as np
class Smoother:
    def __init__(self, window=5):
        self.window = window
        self.values = []
    
    def update(self, val):
        self.values.append(val)
        if len(self.values) > self.window:
            self.values.pop(0)
        return np.mean(self.values)
