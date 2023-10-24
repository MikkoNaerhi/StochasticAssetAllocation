from pyomo.environ import *
M = 1000
model = AbstractModel()


# ----------------------------------
# ----------- PARAMETERS -----------
# ----------------------------------

# Basic Parameters
model.N = Param(within=NonNegativeIntegers) # Number of assets
model.T = Param(within=NonNegativeIntegers) # Number of time points
model.S = Param(within=NonNegativeIntegers) # Number of scenarios

# Financial Parameters
model.SR = Param(within=NonNegativeReals) # Required solvency ratio
model.alpha = Param(within=NonNegativeReals) # Confidence level
model.p_mean = Param(within=NonNegativeReals)
model.w = Param(within=NonNegativeReals) # Fund Weight
model.i0 = Param(within=NonNegativeReals) # Fund rate
model.l = Param(within=NonNegativeReals) # Return linkage
model.L0 = Param(within=NonNegativeReals) # Initial liability
model.sr0 = Param(within=NonNegativeReals) # Initial solvency ratio

# ------------------------
# --------  SETS  --------
# ------------------------

model.i = RangeSet(1, model.N) # Asset indexer
model.t = RangeSet(1, model.T) # Time indexer
model.s = RangeSet(1, model.S) # Scenario indexer

# Simulation params
model.pr = Param(model.s) # Probability of scenario s
model.R = Param(model.i, model.t, model.s) # Return of asset i at time t at scenario s
model.cR = Param(model.i, model.t, model.s) # Cumulative return of asset i at time t at scenario s

# ----------------------------------
# ----------- VARIABLES ------------
# ----------------------------------

model.X0 = Var(model.i, model.s, domain=NonNegativeReals) # Initial amount of asset i
model.X = Var(model.i, model.t, model.s, domain=NonNegativeReals) # Amount of asset i to hold in time t and scenario s
model.a0 = Var(model.i, domain=NonNegativeReals)
model.L = Var(model.t, model.s) # Liabilities
model.sr = Var(model.t, model.s) # Solvency ratio
model.p = Var(model.t, model.s, bounds=(-2,2)) # Supplementary basis
model.b16 = Var(model.t, model.s, bounds=(-0.633,0.214)) # Supplementary factor
model.b16_rsv = Var(model.t, model.s) # Monthly rsv addition from b16
model.C = Var(model.t, model.s, domain=Binary) # Binary variable for solvency requirement. C = 1, if not fulfilled and 0 otherwise
model.j_rsv = Var(model.t, model.s) # Stock return provision at time t and scenario s 
model.i_rsv = Var(model.t, model.s) # Interest provisions at time t and scenario s

# -----------------------
# ------ OBJECTIVE ------
# -----------------------

def obj_expression(m: AbstractModel):
    """ Objective function is the expected value of portfolio at time T
    """
    return sum(m.pr[s] * m.X[i,m.T,s] for i in m.i for s in m.s)

model.OBJ = Objective(rule=obj_expression, sense=maximize)

# -------------------------
# ------ CONSTRAINTS ------
# -------------------------

def asset_amount_rule0(m: AbstractModel, s: int) -> Constraint:
    return sum(m.X0[i,s] for i in m.i) == m.sr0 * m.L0 # Assets equal solvency ratio times initial liability

model.asset_amount_rule0 = Constraint(model.s, rule=asset_amount_rule0)

def asset_amount_rule02(m: AbstractModel, i, s):
    return m.X0[i,s] == sum(m.X0[j,s] for j in m.i) * m.a0[i] # Initial asset allocation is the same for all scenarios

model.asset_amount_rule02 = Constraint(model.i, model.s, rule=asset_amount_rule02)

def asset_amount_rule(m: AbstractModel, i, t, s):
    if t == 1: # Don't rebalance portfolio
        return m.X[i,t,s] == (1+m.R[i,t,s]) * m.X0[i,s]
    else:
        return sum(m.X[j,t,s] for j in m.i) == sum((1+m.R[j,t,s])*m.X[j,t-1,s] for j in m.i) # Assets equal assets in previous time-step + returns

model.asset_amount_rule = Constraint(model.i, model.t, model.s, rule=asset_amount_rule)

#-----------------------------------------
# ----- LIABILITY CALCULATIONS - B16 -----
#-----------------------------------------

pdata=[-2, 0.198, 0.218, 2]
b16data=[-0.633, 0, 0, 0.214]
model.b16_rule = Piecewise(model.t, model.s, model.b16, model.p, pw_pts=pdata, pw_constr_type='EQ', f_rule=b16data, pw_repn='SOS2')

def solvency_ratio_rule(m: AbstractModel, t, s):
    return m.sr[t,s] * m.L[t,s] == sum(m.X[i,t,s] for i in m.i)

model.solvency_ratio_rule = Constraint(model.t, model.s, rule=solvency_ratio_rule)

def p_rule(m: AbstractModel, t, s):
    if t > 1:
        return m.p[t,s] == (1 - m.w) * m.p_mean + m.w * (m.sr[t-1,s] - 1)
    else:
        return m.p[t,s] == (1 - m.w) * m.p_mean + m.w * (m.sr0 - 1)
    
model.p_rule = Constraint(model.t, model.s, rule=p_rule)
    
def b16_rsv_rule(m, t, s):
    if t == 1:
        return m.b16_rsv[t,s] * 4 == m.b16[t,s] * m.L0
    else:
        return (m.b16_rsv[t,s] - m.b16_rsv[t-1,s]) * 4 == m.b16[t,s] * m.L0
    
model.b16_rsv_rule = Constraint(model.t, model.s, rule=b16_rsv_rule)
    
#----------------------------------------
# ------LIABILITY CALCULATIONS - j ------
#----------------------------------------

def j_rsv_rule(m: AbstractModel, t, s):
    return m.j_rsv[t,s] == m.l * (m.cR[1,t,s] - 0.01 * t / 4) * m.L0

model.j_rsv_rule = Constraint(model.t, model.s, rule=j_rsv_rule)

#------------------------------------------
# ------ LIABILITY CALCULATIONS - i0 ------
#------------------------------------------

def i_rsv_rule(m: AbstractModel, t, s):
    return m.i_rsv[t,s] == (m.i0 * t / 4) * m.L0

model.i_rsv_rule = Constraint(model.t, model.s, rule=i_rsv_rule)

def liability_calculation_rule(m: AbstractModel, t, s):
    return m.L[t,s] == m.j_rsv[t,s] + m.i_rsv[t,s] + m.L0 + m.b16_rsv[t,s]

model.liability_calculation_rule = Constraint(model.t, model.s, rule=liability_calculation_rule)
    
#-----------------------------------
# ------ SOLVENCY REQUIREMENT ------
#-----------------------------------

def solvency_indicator_rule(m: AbstractModel, t, s):
    return m.SR - m.sr[t,s] <= M * m.C[t,s]

model.solvency_indicator_rule = Constraint(model.t, model.s, rule=solvency_indicator_rule)

def solvency_probability_rule(m: AbstractModel, t):
    return sum(m.C[t,s] for s in m.s) <= m.alpha * m.S

model.solvency_probability_rule = Constraint(model.t, rule=solvency_probability_rule)


