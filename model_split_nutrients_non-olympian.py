# %load model_split_nutrients_non-olympian.py

"""
Created on Wed Mar 31 18:49:19 2021

@author: Emma Tarmey
"""

import logging
import pandas as pd
import numpy as np
from pyomo.environ import *


model = AbstractModel()
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


# Food and Nutrient data sets
model.Foods     = Set()
model.Nutrients = Set()
model.Days      = Set()


# Model parameters
model.costs               = Param(model.Foods, within = PositiveReals)

model.amountCalories      = Param(model.Foods, within = NonNegativeReals)
model.amountFat           = Param(model.Foods, within = NonNegativeReals)
model.amountSaturates     = Param(model.Foods, within = NonNegativeReals)
model.amountCarbohydrates = Param(model.Foods, within = NonNegativeReals)
model.amountSugar         = Param(model.Foods, within = NonNegativeReals)
model.amountProtein       = Param(model.Foods, within = NonNegativeReals)
model.amountSalt          = Param(model.Foods, within = NonNegativeReals)
model.amountFibre         = Param(model.Foods, within = NonNegativeReals)

 # Minimum nutrient requirements
model.minCalories      = Param(within = NonNegativeReals, default = 0.0)
model.minFat           = Param(within = NonNegativeReals, default = 0.0)
model.minSaturates     = Param(within = NonNegativeReals, default = 0.0)
model.minCarbohydrates = Param(within = NonNegativeReals, default = 0.0)
model.minSugar         = Param(within = NonNegativeReals, default = 0.0)
model.minProtein       = Param(within = NonNegativeReals, default = 0.0)
model.minSalt          = Param(within = NonNegativeReals, default = 0.0)
model.minFibre         = Param(within = NonNegativeReals, default = 0.0)

 # Minimum nutrient requirements
model.maxCalories      = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxFat           = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxSaturates     = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxCarbohydrates = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxSugar         = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxProtein       = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxSalt          = Param(within = NonNegativeReals, \
                               default = float("inf"))
model.maxFibre         = Param(within = NonNegativeReals, \
                               default = float("inf"))


# Model Variables
# Measures the amount of a given food to eat on a given day
model.x = Var(model.Foods, within = NonNegativeIntegers)


# Objective function
# The model seeks to minimise the value of this function
def objective_function(model):
    return (len(model.Days) * sum( (model.costs[i] * model.x[i]) \
                                  for i in model.Foods ) )


# Nutrient constraints
def calories_constraint(model):
    minimum   = model.minCalories
    maximum   = model.maxCalories
    total = sum(model.amountCalories[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def fat_constraint(model):
    minimum   = model.minFat
    maximum   = model.maxFat
    total = sum(model.amountFat[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def saturates_constraint(model):
    minimum   = model.minSaturates
    maximum   = model.maxSaturates
    total = sum(model.amountSaturates[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def carbohydrates_constraint(model):
    minimum   = model.minCarbohydrates
    maximum   = model.maxCarbohydrates
    total = sum(model.amountCarbohydrates[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def sugar_constraint(model):
    minimum   = model.minSugar
    maximum   = model.maxSugar
    total = sum(model.amountSugar[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def protein_constraint(model):
    minimum   = model.minProtein
    maximum   = model.maxProtein
    total = sum(model.amountProtein[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def salt_constraint(model):
    minimum   = model.minSalt
    maximum   = model.maxSalt
    total = sum(model.amountSalt[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def fibre_constraint(model):
    minimum   = model.minFibre
    maximum   = model.maxFibre
    total = sum(model.amountFibre[i] * model.x[i] for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied


# Attach all above methods to the pyomo model object
model.cost = Objective(rule = objective_function)

model.calories      = Constraint(rule = calories_constraint)
model.fat           = Constraint(rule = fat_constraint)
model.saturates     = Constraint(rule = saturates_constraint)
model.carbohydrates = Constraint(rule = carbohydrates_constraint)

model.sugar         = Constraint(rule = sugar_constraint)
model.protein       = Constraint(rule = protein_constraint)
model.salt          = Constraint(rule = salt_constraint)
model.fibre         = Constraint(rule = fibre_constraint)
