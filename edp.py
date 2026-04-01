from loss import LossEquation

class Generator:
    def __init__(self, a: float, b: float, c: float, Pmin: float, Pmax: float):
        self.a = a
        self.b = b
        self.c = c
        
        self.Pmin = Pmin
        self.Pmax = Pmax
        
        self.P = 0 # Output Power
        self.Pe = 0 # Exceed Power
        self.is_locked = False # Limit Lock
    
    def C(self):
        return self.a + self.b*self.P + self.c*(self.P**2)


class Grid:
    def __init__(self, Pd: float, G: list[Generator], loss: LossEquation):
        """
        :param Pd: Dispatch Power [MW]
        :param G: List of Generator class
        :param loss: Loss Equation [MW]
        """
        
        self.Pd = Pd # Dispatch Power [MW]
        self.G = G # Generators
        self.MC = 0 # Marginal Cost [$/MWh]
        self.loss = loss # Loss Equation [MW]
        
        self.j = len(G)
    
    @property
    def NLG(self):
        return [gen for gen in self.G if not getattr(gen, 'is_locked', False)]

    @property
    def free_idx(self):
        return [i for i, gen in enumerate(self.G) if not getattr(gen, 'is_locked', False)]
    
    def Pg(self) -> list[float]: # Opitmal Output Power
        return [gen.P for gen in self.G]
    
    def TC(self) -> float: # Total Cost
        return sum([gen.C() for gen in self.G])
    
    def KKT(self) -> bool: # Karush-Kuhn-Tucker conditions
        violation_gen = []
        
        for gen in self.G:
            if not getattr(gen, 'is_locked', False): 
                if gen.P < gen.Pmin:
                    gen.Pe = gen.Pmin - gen.P
                    violation_gen.append(gen)
                elif gen.P > gen.Pmax:
                    gen.Pe = gen.P - gen.Pmax
                    violation_gen.append(gen)

        if not violation_gen:
            return False

        locked_gen = max(violation_gen, key=lambda g: g.Pe)

        if locked_gen.P < locked_gen.min:
            locked_gen.P = locked_gen.min
        else:
            locked_gen.P = locked_gen.max
            
        locked_gen.is_locked = True
        
        return True