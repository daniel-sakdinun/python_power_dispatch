import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fourpace.psys import Grid, CEP
from fourpace.pfa import plan
import pandas as pd

grid = Grid.load('config.yaml')
load_profile = pd.read_csv('profile.csv')

plan(grid, load_profile, relax='SOCP', solver='CLARABEL')