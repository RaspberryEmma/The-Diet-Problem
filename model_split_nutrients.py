# %load model_split_nutrients.py

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

model.training            = Param(model.Days, within = Boolean)

model.amountCalories      = Param(model.Foods, within = NonNegativeReals)
model.amountFat           = Param(model.Foods, within = NonNegativeReals)
model.amountSaturates     = Param(model.Foods, within = NonNegativeReals)
model.amountCarbohydrates = Param(model.Foods, within = NonNegativeReals)
model.amountSugar         = Param(model.Foods, within = NonNegativeReals)
model.amountProtein       = Param(model.Foods, within = NonNegativeReals)
model.amountSalt          = Param(model.Foods, within = NonNegativeReals)
model.amountFibre         = Param(model.Foods, within = NonNegativeReals)

 # Minimum nutrient requirements when not training
model.minCalories      = Param(within = NonNegativeReals, default = 0.0)
model.minFat           = Param(within = NonNegativeReals, default = 0.0)
model.minSaturates     = Param(within = NonNegativeReals, default = 0.0)
model.minCarbohydrates = Param(within = NonNegativeReals, default = 0.0)
model.minSugar         = Param(within = NonNegativeReals, default = 0.0)
model.minProtein       = Param(within = NonNegativeReals, default = 0.0)
model.minSalt          = Param(within = NonNegativeReals, default = 0.0)
model.minFibre         = Param(within = NonNegativeReals, default = 0.0)

 # Minimum nutrient requirements when training
model.minCaloriesTrain      = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minFatTrain           = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minSaturatesTrain     = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minCarbohydratesTrain = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minSugarTrain         = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minProteinTrain       = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minSaltTrain          = Param(within = NonNegativeReals, \
                                    default = 0.0)
model.minFibreTrain         = Param(within = NonNegativeReals, \
                                    default = 0.0)

 # Maximum nutrient requirements when not training
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

 # Maximum nutrient requirements when training
model.maxCaloriesTrain      = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxFatTrain           = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxSaturatesTrain     = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxCarbohydratesTrain = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxSugarTrain         = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxProteinTrain       = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxSaltTrain          = Param(within = NonNegativeReals, \
                                    default = float("inf"))
model.maxFibreTrain         = Param(within = NonNegativeReals, \
                                    default = float("inf"))


# Model Variable
# Measures the amount of a given food to eat on a given day
model.x = Var(model.Days, model.Foods, within = NonNegativeIntegers)


# Objective function
# the model seeks to minimise the value of this function
def objective_function(model):
    total = sum( sum( (model.costs[i] * model.x[d, i]) \
                     for i in model.Foods) for d in model.Days )
    return total


# Nutrient constraints
def calories_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minCaloriesTrain
        maximum = model.maxCaloriesTrain
    else:
        minimum = model.minCalories
        maximum = model.maxCalories
    
    total = sum(model.amountCalories[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def fat_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minFatTrain
        maximum = model.maxFatTrain
    else:
        minimum = model.minFat
        maximum = model.maxFat
    
    total = sum(model.amountFat[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def saturates_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minSaturatesTrain
        maximum = model.maxSaturatesTrain
    else:
        minimum   = model.minSaturates
        maximum = model.maxSaturates
    
    total = sum(model.amountSaturates[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def carbohydrates_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minCarbohydratesTrain
        maximum = model.maxCarbohydratesTrain
    else:
        minimum = model.minCarbohydrates
        maximum = model.maxCarbohydrates
    
    total = sum(model.amountCarbohydrates[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def sugar_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minSugarTrain
        maximum = model.maxSugarTrain
    else:
        minimum = model.minSugar
        maximum = model.maxSugar
    
    total = sum(model.amountSugar[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def protein_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minProteinTrain
        maximum = model.maxProteinTrain
    else:
        minimum = model.minProtein
        maximum = model.maxProtein
    
    total = sum(model.amountProtein[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def salt_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minSaltTrain
        maximum = model.maxSaltTrain
    else:
        minimum = model.minSalt
        maximum = model.maxSaltTrain
    
    total = sum(model.amountSalt[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied

def fibre_constraint(model, d):
    # If today is a training day, raise our requirements appropriately
    if ( model.training[d] ):
        minimum = model.minFibreTrain
        maximum = model.maxFibreTrain
    else:
        minimum = model.minFibre
        maximum = model.maxFibre
    
    total = sum(model.amountFibre[i] * model.x[d, i] \
                for i in model.Foods)
    satisfied = (total >= minimum) and (total <= maximum)
    return satisfied


# Attach all above methods to the pyomo model object
model.cost = Objective(rule = objective_function)

model.calories      = Constraint(model.Days, rule = calories_constraint)
model.fat           = Constraint(model.Days, rule = fat_constraint)
model.saturates     = Constraint(model.Days, \
                                 rule = saturates_constraint)
model.carbohydrates = Constraint(model.Days, \
                                 rule = carbohydrates_constraint)

model.sugar         = Constraint(model.Days, rule = sugar_constraint)
model.protein       = Constraint(model.Days, rule = protein_constraint)
model.salt          = Constraint(model.Days, rule = salt_constraint)
model.fibre         = Constraint(model.Days, rule = fibre_constraint)
