import numpy as np

def CSD(x, d):  #Compression/tension spring design (CSD)
    PCONST = 100  # % PENALTY FUNCTION CONSTANT

    fit = (x[2] + 2) * x[1] * (x[0] ** 2)
    G1 = 1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4)
    G2 = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2)
    G3 = 1 - (140.45 * x[0]) / (x[1] ** 2 * x[2])
    G4 = ((x[0] + x[1]) / 1.5) - 1

    PHI = fit + PCONST * (max(0, G1) ** 2 + max(0, G2) ** 2 + max(0, G3) ** 2 + max(0, G4) ** 2) # % PENALTY FUNCTION
    return PHI

def WBD(x, d): # Welded Beam Design (WBD)
    P = 6000  # % APPLIED TIP LOAD
    E = 30e6  # % YOUNGS MODULUS OF BEAM
    G = 12e6  # % SHEAR MODULUS OF BEAM
    tem = 14  # % LENGTH OF CANTILEVER PART OF BEAM
    PCONST = 100000  # % PENALTY FUNCTION CONSTANT
    TAUMAX = 13600  # % MAXIMUM ALLOWED SHEAR STRESS
    SIGMAX = 30000  # % MAXIMUM ALLOWED BENDING STRESS
    DELTMAX = 0.25  # % MAXIMUM ALLOWED TIP DEFLECTION
    M = P * (tem + x[1] / 2)  # % BENDING MOMENT AT WELD POINT
    R = np.sqrt((x[1] ** 2) / 4 + ((x[0] + x[2]) / 2) ** 2)  # % SOME CONSTANT
    J = 2 * (np.sqrt(2) * x[0] * x[1] * ((x[1] ** 2) / 4 + ((x[0] + x[2]) / 2) ** 2))  # % POLAR MOMENT OF INERTIA
    # print(M, R, J)
    PHI = 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])  # % OBJECTIVE FUNCTION
    SIGMA = (6 * P * tem) / (x[3] * x[2] ** 2)  # BENDING STRESS
    DELTA = (4 * P * tem ** 3) / (E * x[2] ** 3 * x[3])  # TIP DEFLECTION
    PC = 4.013 * E * np.sqrt((x[2] ** 2 * x[3] ** 6) / 36) * (1 - x[2] * np.sqrt(E / (4 * G)) / (2 * tem)) / (
                tem ** 2)  # % BUCKLING LOAD
    TAUP = P / (np.sqrt(2) * x[0] * x[1])  # % 1 ST DERIVATIVE OF SHEAR STRESS
    TAUPP = (M * R) / J  # % 2ND DERIVATIVE OF SHEAR STRESS
    TAU = np.sqrt(TAUP ** 2 + 2 * TAUP * TAUPP * x[1] / (2 * R) + TAUPP ** 2)  # % SHEAR STRESS
    G1 = TAU - TAUMAX  # % MAX SHEAR STRESS CONSTRAINT
    G2 = SIGMA - SIGMAX  # ; % MAX BENDING STRESS CONSTRAINT
    # % G3 = L(1) - L(4); % WELD COVERAGE CONSTRAINT
    G3 = DELTA - DELTMAX
    G4 = x[0] - x[3]
    G5 = P - PC
    G6 = 0.125 - x[0]
    # %G4 = 0.10471 * L(1) ^ 2 + 0.04811 * L(3) * L(4) * (14 + L(2)) - 5 # % MAX COST CONSTRAINT
    # % G5 = 0.125 - L(1); % MAX WELD THICKNESS CONSTRAINT
    # % G6 = DELTA - DELTMAX;
    # % MAX TIP DEFLECTION CONSTRAINT
    # % G7 = P - PC; % BUCKLING LOAD CONSTRAINT
    G7 = 1.10471 * x[0] ** 2 + 0.04811 * x[2] * x[3] * (14 + x[1]) - 5  # ;
    PHI = PHI + PCONST * (max(0, G1) ** 2 + max(0, G2) ** 2 + max(0, G3) ** 2 + max(0, G4) ** 2 + max(0, G5) ** 2 + max(0,G6) ** 2 + max(0, G7) ** 2)  # % PENALTY FUNCTION

    return PHI

def PVD(x, d):  #Pressure Vessel Design
    PCONST = 10000# % PENALTY FUNCTION CONSTANT
    fit = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] ** 2 + 3.1661 * x[0] ** 2 * x[3] + 19.84 * x[0] ** 2 * x[2]
    G1 = -x[0] + 0.0193 * x[2]
    G2 = -x[2] + 0.00954 * x[2]
    G3 = -np.pi * x[2] ** 2 * x[3] - (4 / 3) * np.pi * x[2] ** 3 + 1296000
    G4 = x[3] - 240
    PHI = fit + PCONST * (max(0, G1) ** 2 + max(0, G2) ** 2 + max(0, G3) ** 2 + max(0, G4) ** 2) # % PENALTY FUNCTION

    return PHI

def HNO(x, d):    # Himmelblau's Nonlinear Optimisation
    PCONST = 3.5
    fit = 5.3578547* (x[2] ** 2) + 0.8356891 * (x[0] * x[4]) + 37.293239 * (x[0]) - 40792.141
    G1 = min( 85.334407 + 0.0056858 * x[1] * x[4] + 0.00026 * x[0] * x[3] - 0.0022053 * x[2] * x[4], 92)
    G2 = min( 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * (x[2] ** 2), 110)
    G3 = min( 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3], 25)
    PHI = fit + PCONST*((max(0,G1)**2) + (max(90,G2)**2) + (max(20,G3)**2))
    
    return PHI

def SRP(x, d):     # Speed Reducer Problem
    PCONST = 1000
    fit = (0.7854 * x[0] * x[1]**2 * (3.3333 * x[2]**2 - 14.9334 * x[2] - 43.0934) 
       - 1.508 * x[0] * (x[5]**2 + x[6]**2) 
       + 7.4777 * (x[5]**3 + x[6]**3) 
       + 0.7854 * (x[3] * x[5]**2 + x[4] * x[6]**2))

    # Calculate the constraints
    G1 = 27 / (x[0] * x[1]**2 * x[2]) - 1
    G2 = 397.5 / (x[0] * x[1]**2 * x[2]**2)
    G3 = 1.93 * x[3]**3 / (x[1] * x[2] * x[5]**4)
    G4 = 1.93 * x[4]**3 / (x[1] * x[2] * x[6]**4)
    G5 = 1.0 / (110 * x[5]**3) * np.sqrt((745.0 * x[3] / (x[1] * x[2]))**2 + 16.9e6) - 1
    G6 = 1.0 / (85 * x[6]**3) * np.sqrt((745.0 * x[4] / (x[1] * x[2]))**2 + 157.5e6) - 1
    G7 = x[1] * x[2] / 40
    G8 = 5 * x[1] / x[0] - 1
    G9 = x[0] / (12 * x[1]) - 1
    G10 = (1.5 * x[5] + 1.9) / x[3] - 1
    G11 = (1.1 * x[6] + 1.9) / x[4] - 1

    # Calculate the penalty function
    PHI = (fit + PCONST * (max(0, G1)**2 + max(0, G2)**2 +
                        max(0, G3)**2 + max(0, G4)**2 +
                        max(0, G5)**2 + max(0, G6)**2 +
                        max(0, G7)**2 + max(0, G8)**2 +
                        max(0, G9)**2 + max(0, G10)**2 +
                        max(0, G11)**2))

    return PHI

def CBP(x, d):     # Cantilever Beam Problem
    PCONST = 1
    
    fit = 0.0624 * (x[0] + x[1] + x[2] + x[3] + x[4])
    
    G1_1 = 61/((x[0]) ** 3) if (x[0]) ** 3 != 0 else 1
    G1_2 = 37/((x[1]) ** 3) if (x[1]) ** 3 != 0 else 1
    G1_3 = 19/((x[2]) ** 3) if (x[2]) ** 3 != 0 else 1
    G1_4 = 7/((x[3]) ** 3) if (x[3]) ** 3 != 0 else 1
    G1_5 = 1/((x[4]) ** 3) if (x[4]) ** 3 != 0 else 1
    G1 = G1_1 + G1_2 + G1_3 + G1_4 + G1_5 - 1
    
    PHI = fit + PCONST * (max(G1, 0)**2)
    
    return PHI

def GTD(x, d):     # Gear Train Design
    import math
    fit = ((1 / 6.931) - math.floor(x[0]) * math.floor(x[1]) / (math.floor(x[2]) * math.floor(x[3])))**2
    PHI = fit
    return PHI

def DCB(x, d):     # Multiple Disc Clutch Brake Problem
    PCONST = 1     # Penalty function constant
    DR = 20
    L = 30
    VSR = 10
    MU = 0.5
    S = 1.5
    MS = 40
    MF = 3
    n = 250
    P = 1
    I = 55
    T = 15

    MH = (2/3) * MU * x[3] * x[4] * ((x[1]**3 - x[0]**3) / (x[1]**2 - x[0]**2))
    PRZ = x[3] / np.pi * (x[1]**2 - x[0]**2)
    VSR1 = (2 / 90) * np.pi * n * ((x[1]**3 - x[0]**3) / (x[1]**2 - x[0]**2))
    T1 = (I * np.pi * n) / (30 * (MH + MF))

    fit = np.pi * x[2] * (x[1]**2 - x[0]**2) * (x[4] + 1) * 8 * P

    G1 = x[1] - x[0] - DR
    G2 = L - (x[4] + 1) * x[2]
    G3 = P - PRZ
    G4 = P * VSR - PRZ * VSR
    G5 = VSR - VSR1
    G6 = T - T1
    G7 = MH - S * MS
    G8 = T1

    PHI = fit + PCONST * (max(0, G1)**2 + max(0, G2)**2 +
                        max(0, G3)**2 + max(0, G4)**2 +
                        max(0, G5)**2 + max(0, G6)**2 +
                        max(0, G7)**2 + max(0, G8)**2)
    
    return PHI