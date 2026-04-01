from abc import ABC, abstractmethod
import numpy as np

class LossEquation(ABC):
    def __init__(self):
        self.Pg = None 

    @abstractmethod
    def Pl(self) -> float:
        pass

    @abstractmethod
    def gradient(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def hessian(self) -> np.ndarray:
        pass

    def penalty(self) -> np.ndarray:
        return 1 / (1 - self.gradient())

class Lossless(LossEquation):
    def __init__(self, num_gen: int):
        super().__init__()
        self.Pg = np.zeros(num_gen)
        
    def Pl(self) -> float:
        return 0.0
        
    def gradient(self) -> np.ndarray:
        return np.zeros_like(self.Pg)
    
    def hessian(self) -> np.ndarray:
        return np.zeros((len(self.Pg), len(self.Pg)))

class KronLoss(LossEquation):
    def __init__(self, num_gen:int , B: np.ndarray, B0: np.ndarray, B00: float):
        super().__init__()
        self.Pg = np.zeros(num_gen)
        self.B = B
        self.B0 = B0
        self.B00 = B00
    
    def Pl(self) -> float:
        return (self.Pg.T @ self.B @ self.Pg) + (self.B0 @ self.Pg) + self.B00
    
    def gradient(self) -> np.ndarray:
        return 2 * self.B @ self.Pg + self.B0
    
    def hessian(self) -> np.ndarray:
        return 2 * self.B


class DCFlowLoss(LossEquation):
    def __init__(self, num_gen: int, PTDF: np.ndarray, R_lines: np.ndarray, S_base: float = 100.0):
        super().__init__()
        self.Pg = np.zeros(num_gen)
        self.PTDF = PTDF
        self.R = np.diag(R_lines) 
        self.S_base = S_base
        
        self.H = self.PTDF.T @ self.R @ self.PTDF

    def Pl(self) -> float:
        Pg_pu = self.Pg / self.S_base
        loss_pu = Pg_pu.T @ self.H @ Pg_pu
        return float(loss_pu * self.S_base)

    def gradient(self) -> np.ndarray:
        Pg_pu = self.Pg / self.S_base
        grad = 2 * self.H @ Pg_pu
        return grad
    
    def hessian(self) -> np.ndarray:
        return (2 * self.H) / self.S_base