import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


class DataDescription:
    '''An object that holds descriptive values of some dataset
    
    '''
    def __init__(
            self,
            X: jnp.ndarray = None,
            y: jnp.ndarray = None,
            y_pred: jnp.ndarray = None,
            uncertainty: dict = None 
    ):
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.uncertainty = uncertainty