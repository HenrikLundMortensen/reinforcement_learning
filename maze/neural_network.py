import numpy as np
import tensorflow as tf


class NeuralNet():
    """
    """
    
    def __init__(self):
        """
        """
        self.n_hidden = 100
        self.nn = self._infer()
        

    def _infer(self):
        """
        """
    
        pos = tf.placeholder(tf.int32,shape=[2,1])
        layer1 = tf.contrib.layers.fully_connected(inputs=pos,num_outputs=self.n_hidden)
        out = tf.contrib.layers.fully_connected(layer1,num_outputs=4,activation_fn=None)
        return out

    
    def loss(self,pos):
        """
        """

        
        
        
        
    

    
