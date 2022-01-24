import pyomo.environ as pyo

# Sum over all j of (m_j * x_j)
def obj_expression(m):
    return pyo.summation(m.c, m.x)

# Returns true / false
def ax_constraint_rule(m, i):
    return (sum((m.a[i, j] * m.x[j]) for j in m.J) >= m.b[i])

# Basic LP
def concrete_model():
    model             = pyo.ConcreteModel()
    model.x           = pyo.Var([1, 2], domain = pyo.NonNegativeReals)
    model.OBJ         = pyo.Objective(expr = ((2 * model.x[1]) + (3 * model.x[2])))
    model.Constraint1 = pyo.Constraint(expr = (3 * model.x[1]) + (4 * model.x[2]) >= 1)
    return model

# Abstract LP
# More similar to diet problem
def abstract_model():
    model   = pyo.AbstractModel()

    model.m = pyo.Param(within = pyo.NonNegativeIntegers)
    model.n = pyo.Param(within = pyo.NonNegativeIntegers)

    model.I = pyo.RangeSet(1, model.m)
    model.J = pyo.RangeSet(1, model.n)

    model.a = pyo.Param(model.I, model.J)
    model.b = pyo.Param(model.I)
    model.c = pyo.Param(model.J)

    model.x             = pyo.Var(model.J, domain = pyo.NonNegativeReals)
    model.OBJ           = pyo.Objective(rule = obj_expression)
    model.AxbConstraint = pyo.Constraint(model.I, rule = ax_constraint_rule)
    return model

def pyomo_create_model():
    return abstract_model()

