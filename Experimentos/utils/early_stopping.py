import torch
import numpy as np

class EarlyStopping():
    def __init__(self, patience=5):
        self.best_test_loss = np.inf
        self.delta = 0.001
        self.best_model_state = None
        self.patience = patience
        self.counter = 0
    
    def __call__(self, test_loss, model):
        early_stop = False
        if test_loss < self.best_test_loss - self.delta:
            self.best_test_loss = test_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            print(f"Patience: {self.counter} out of {self.patience}")
            if self.counter == self.patience:
                early_stop = True
        return early_stop

    def save_model(self, model_path):
        torch.save(self.best_model_state, model_path)
        print(f"Model saved at {model_path}")