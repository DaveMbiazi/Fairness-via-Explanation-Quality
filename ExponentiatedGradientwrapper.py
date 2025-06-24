import tensorflow as tf
import numpy as np

class ExpoGrad():
    def __init__(self, expgrad_model):
        self.expgrad_model = expgrad_model

    def call(self, inputs, training = False):
        return self.expgrad_model.predict(inputs)

    def __call__(self, inputs, training = False):
        preds = self.expgrad_model.predict(inputs)
        return np.float64(preds)
    
    def evaluate(self, inputs, targets):
        loss = tf.metrics.binary_crossentropy(targets, self(inputs))

        metric = {'loss': loss}
        return metric
    
    def predict(self, inputs):
        return self(inputs) 