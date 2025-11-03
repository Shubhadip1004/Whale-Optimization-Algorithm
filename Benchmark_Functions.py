import numpy as np

# Benchmark Functions for Optimization Algorithms

def sphere(pos, dimensions):
    """Sphere Function

    The Sphere function is a simple benchmark function for optimization.
    It is defined as the sum of the squares of its input variables.

    Global Minimum:
        f(0, 0, ..., 0) = 0
        
    Dimension Range: 3 to 256 (commonly used: 30)
    Common Domain: [-5.12, 5.12]
    """
    return sum(x**2 for x in pos)

def rosenbrock(pos, dimensions):
    """Rosenbrock Function

    The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms.
    It is characterized by a narrow, curved valley.

    Global Minimum:
        f(1, 1, ..., 1) = 0
    Dimension Range: 5 to 100 (commonly used: 30)
    Common Domain: [-5, 10]"""
    
    res = 0
    for i in range(dimensions-1):
        res += (pos[i] - 1)**2 + 100 * (pos[i+1] + pos[i]**2)**2
    
    return res

def quartic(pos, dimensions):
    """Quartic Function without Noise

    The Quartic function without noise is a benchmark function for optimization that includes a deterministic component.
    It is defined as the sum of the fourth powers of its input variables.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-1.28, 1.28]"""
    res = 0
    for i in range(dimensions):
        res += (i + 1) * (pos[i] ** 4)
    
    return res

def step(pos, dimensions):
    """Step Function

    The Step function is a simple benchmark function for optimization.
    It is defined as the sum of the squares of the ceiling of its input variables.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 5 to 100 (commonly used: 30)
    Common Domain: [-100, 100]"""

    return sum(int(x + 0.5)**2 for x in pos)

def schwefel_2point22(pos, dimensions):
    """Schwefel 2.22 Function

    The Schwefel 2.22 function is a benchmark function for optimization.
    It is defined as the sum of the absolute values of its input variables plus the product of the absolute values of its input variables.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-10, 10]"""
    sum_abs = sum(abs(x) for x in pos)
    prod_abs = 1
    for x in pos:
        prod_abs *= abs(x)
    
    return sum_abs + prod_abs

def sumsquare(pos, dimensions):
    """Sum Square Function

    The Sum Square function is a benchmark function for optimization.
    It is defined as the sum of the squares of its input variables multiplied by their respective indices.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-10, 10]"""
    res = 0
    for i in range(dimensions):
        res += (i + 1) * (pos[i] ** 2)
    
    return res

def elliptic(pos, dimensions):
    """Elliptic Function

    The Elliptic function is a benchmark function for optimization.
    It is defined as the sum of the squares of its input variables, each multiplied by a scaling factor that increases exponentially with the index.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-100, 100]"""
    res = 0
    for i in range(dimensions):
        res += (pos[i] ** 2) * (10**6)**(i / (dimensions - 1))
    
    return res

def rastrigin(pos, dimensions):
    """Rastrigin Function

    The Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is characterized by a large search space and many local minima.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-5.12, 5.12]"""
    res = 0
    for i in range(dimensions):
        res += 10 + pos[i]**2 - 10 * np.cos(2 * np.pi * pos[i])
    
    return res

def griewank(pos, dimensions):
    """Griewank Function

    The Griewank function is a complex benchmark function for optimization.
    It is defined as the sum of the squares of its input variables divided by 4000 minus the product of the cosines of its input variables divided by the square root of their indices, plus one.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-600, 600]"""

    res = sum(x**2 / 4000 for x in pos)
    prod = 1
    for i, x in enumerate(pos):
        prod *= np.cos(x / np.sqrt(i + 1))
    return res - prod + 1

def ackley(pos, dimensions):
    """Ackley Function

    The Ackley function is a widely used benchmark function for optimization.
    It is characterized by a nearly flat outer region and a large hole at the center.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-32, 32]"""

    sum1 = sum(x**2 for x in pos)
    sum2 = sum(np.cos(2 * np.pi * x) for x in pos)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / dimensions))
    term2 = -np.exp(sum2 / dimensions)
    return term1 + term2 + 20 + np.e

def michalewicz(pos, dimensions):
    """Michalewicz Function

    The Michalewicz function is a challenging benchmark function for optimization.
    It is defined as the negative sum of the sine of its input variables multiplied by the sine of the square of its input variables divided by pi, raised to the power of 2m.

    Global Minimum:
        f(2.20, 1.57, ..., 1.57) ≈ -1.8013 for 2D
        f(2.20, 1.57, ..., 1.57) ≈ -4.6876 for 5D
        f(2.20, 1.57, ..., 1.57) ≈ -9.6602 for 10D

    Dimension Range: 2 to 100 (commonly used: 10)
    Common Domain: [0, π]"""
    
    m = 10
    res = 0
    for i in range(dimensions):
        res += np.sin(pos[i]) * (np.sin((i + 1) * pos[i]**2 / np.pi))**(2 * m)
    return -res

def penalized1(pos, dimensions):
    """Penalized Function 1

    The Penalized Function 1 is a benchmark function for optimization that includes penalty terms for constraint violations.
    It is defined as a combination of a base function and penalty terms based on the input variables.

    Global Minimum:
        f(1, 1, ..., 1) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-50, 50]"""

    k = 100
    m = 4
    a = 10
    
    def u(x, a, k, m):
        if x > a:
            return k * (x - a) ** m
        elif x < -a:
            return k * (-x - a) ** m
        else:
            return 0

    def y(x):
        return 1 + (x + 1) / 4

    term1 = 10 * np.sin(np.pi * y(pos[0]))**2
    term2 = sum((y(pos[i]) - 1)**2 * (1 + 10 * np.sin(np.pi * y(pos[i + 1]))**2) for i in range(dimensions - 1))
    term3 = (y(pos[-1]) - 1)**2
    penalty = sum(u(pos[i], a, k, m) for i in range(dimensions))

    res = (term1 + term2 + term3) * np.pi / dimensions 
    return res + penalty

def schwefel(pos, dimensions):
    """Schwefel Function

    The Schwefel function is a complex benchmark function for optimization.
    It is defined as the sum of the negative product of its input variables and the sine of the square root of their absolute values, plus a constant term.

    Global Minimum:
        f(421, 421, ..., 421) = -418.9829 * dimensions
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-500, 500]"""
    
    return sum(-x * np.sin(np.sqrt(abs(x))) for x in pos)
    
def weierstrass(pos, dimensions):
    """Weierstrass Function

    The Weierstrass function is a continuous but nowhere differentiable function used as a benchmark for optimization algorithms.
    It is defined as the sum of cosine terms with exponentially decreasing amplitudes and frequencies.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-0.5, 0.5]"""
    
    a = 0.5
    b = 3
    k_max = 20
    
    sum1 = sum(sum(a**k * np.cos(2 * np.pi * b**k * (pos[i] + 0.5)) for k in range(k_max + 1)) for i in range(dimensions))
    sum2 = dimensions * sum(a**k * np.cos(np.pi * b**k) for k in range(k_max + 1))
    
    return sum1 - sum2

def non_continuous_rastrigin(pos, dimensions):
    """Non-Continuous Rastrigin Function

    The Non-Continuous Rastrigin function is a variant of the Rastrigin function that introduces discontinuities.
    It is defined as the sum of the squares of its input variables (rounded to the nearest integer if greater than 0.5) minus 10 times the cosine of 2π times its input variables, plus 10 times the number of dimensions.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-5.12, 5.12]"""
    
    res = 0
    for i in range(dimensions):
        xi = pos[i]
        if abs(xi) > 0.5:
            xi = round(xi * 2) / 2
        res += xi**2 - 10 * np.cos(2 * np.pi * xi)
    
    return res + 10 * dimensions

def penalized2(pos, dimensions):
    """Penalized Function 2

    The Penalized Function 2 is a benchmark function for optimization that includes penalty terms for constraint violations.
    It is defined as a combination of a base function and penalty terms based on the input variables.

    Global Minimum:
        f(1, 1, ..., 1) = 0
    Dimension Range: 2 to 100 (commonly used: 30)
    Common Domain: [-5.12, 5.12]"""

    k = 100
    m = 4
    a = 5
    
    def u(x, a, k, m):
        if x > a:
            return k * (x - a) ** m
        elif x < -a:
            return k * (-x - a) ** m
        else:
            return 0

    term1 = 0.1 * ((np.sin(3 * np.pi * pos[0]))**2 + sum((pos[i] - 1)**2 * (1 + (np.sin(3 * np.pi * pos[i + 1]))**2) for i in range(dimensions - 1)) + (pos[-1] - 1)**2)
    penalty = sum(u(pos[i], a, k, m) for i in range(dimensions))

    return term1 + penalty

def schwefel_2point26(pos, dimensions):
    """Schwefel 2.26 Function

    The Schwefel 2.26 function is a benchmark function for optimization.
    It is defined as the sum of the input variables multiplied by the sine of the square root of their absolute values.

    Global Minimum:
        f(420.9687, 420.9687, ..., 420.9687) = -418.9829 * dimensions
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-500, 500]"""

    return sum(x * np.sin(np.sqrt(abs(x))) for x in pos) - 418.9829 * dimensions

def schaffer(pos, dimensions):
    """Schaffer Function N. 2

    The Schaffer function N. 2 is a benchmark function for optimization.
    It is defined as 0.5 plus the square of the sine of the sum of the squares of its input variables minus 0.5, divided by (1 + 0.001 times the sum of the squares of its input variables) squared.

    Global Minimum:
        f(0, 0) = 0
    Dimension Range: 2 (fixed)
    Common Domain: [-100, 100]"""

    sum_sq = sum(x**2 for x in pos)
    numerator = (np.sin(sum_sq)**2 - 0.5)
    denominator = (1 + 0.001 * sum_sq)**2
    return 0.5 + numerator / denominator

def alpine(pos, dimensions):
    """Alpine Function

    The Alpine function is a benchmark function for optimization.
    It is defined as the sum of the absolute values of its input variables multiplied by the sine of their square roots.

    Global Minimum:
        f(0, 0, ..., 0) = 0
    Dimension Range: 10 to 100 (commonly used: 30)
    Common Domain: [-10, 10]"""

    return sum(abs(x * np.sin(np.sqrt(abs(x)))) for x in pos)

def himmelblau(pos, dimensions):
    """Himmelblau's Function

    Himmelblau's function is a multi-modal benchmark function for optimization.
    It is defined as the sum of the squares of two expressions involving its input variables.

    Global Minima:
        f(3.0, 2.0) = 0
        f(-2.805118, 3.131312) = 0
        f(-3.779310, -3.283186) = 0
        f(3.584428, -1.848126) = 0

    Dimension Range: 2 (fixed)
    Common Domain: [-6, 6]"""

    x = pos[0]
    y = pos[1]
    term1 = (x**2 + y - 11)**2
    term2 = (x + y**2 - 7)**2
    return term1 + term2