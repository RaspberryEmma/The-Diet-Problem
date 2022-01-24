# %load model.py
"""
Created on Wed Mar 31 18:49:19 2021

@author: Emma Tarmey
"""

import pandas as pd
import numpy as np
from pyomo.environ import *


model = AbstractModel()


# Food and Nutrient data sets
model.Foods     = Set()
model.Nutrients = Set()


# Model parameters
model.costs            = Param(model.Foods, within = PositiveReals)
model.nutrient_amounts = Param(model.Foods, model.Nutrients, within = NonNegativeReals)

model.Nmin = Param(model.Nutrients, within = NonNegativeReals, default = 0.0) # minimum nutrient requirement
model.Nmax = Param(model.Nutrients, within = NonNegativeReals, default = float("inf")) # maximum nutrient requirement


# Model Variable
model.x = Var(model.Foods, within = NonNegativeIntegers) # amount of each food in diet solution


# Objective function - the model seeks to minimise the value of this function for any given input
def objective_function(model):
    total = 0.0
    
    for i in model.Foods:
        total += (model.costs[i] * model.x[i])
    
    return total


def nutrient_constraints(model, j):
	total = sum((model.nutrient_amounts[i, j] * model.x[i]) for i in model.Foods)
	lower = (total >= Nmin[j])
	upper = (total <= Nmax[j])
	return (lower and upper)


model.cost      = Objective(rule = objective_function)
model.nutrients = Constraint(model.Nutrients, rule = nutrient_constraints)
