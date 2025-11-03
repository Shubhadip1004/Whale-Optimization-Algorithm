import numpy as np
import random

global pos, func, l_b, u_b, dimensions

def checkpos(pos, l_b, u_b):
    return np.clip(pos, l_b, u_b)

def dist(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def run_woa(pos, func, l_b, u_b, dimensions, iterations, whale):
    print(">>>>>> Running WOA Algorithm >>>>>")

    b = 1.5
    fx_lst = [func(p, dimensions) for p in pos]
    ylst = []
    
    for it in range(iterations):
        best_pos = pos[np.argmin(fx_lst)]
        best_fx = min(fx_lst)
        
        a = 2 - (2 * it) / (iterations - 1)
        for i in range(whale):
            p = random.random()
            l = random.random()
            A = 2 * a * random.random() - a
            C = 2 * random.random()

            if p < 0.5:
                if abs(A) < 1:
                    D = C * best_pos - pos[i]
                    temp = best_pos - A * D
                    
                else:
                    r = random.randint(0, whale-1)
                    D = C * pos[r] - pos[i]
                    temp = pos[r] - A * D
            else:
                D = dist(best_pos, pos[i])
                temp = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos

            temp = checkpos(temp, l_b, u_b)
            temp_fit = func(temp, dimensions)
            
            if temp_fit < fx_lst[i]:
                pos[i] = temp
                fx_lst[i] = temp_fit
            
        ylst.append(best_fx)
        
    best_pos = pos[np.argmin(fx_lst)]
    best_fx = min(fx_lst)

    compressed_data = ylst + best_pos.tolist() + [best_fx]

    return compressed_data


# Testing
# if __name__ == "__main__":
#     import Benchmark_Functions as functions

#     n_whales = 30
#     dimensions = 30
#     iterations = 500
#     lower_bound = -100
#     upper_bound = 100

#     initial_positions = np.random.uniform(lower_bound, upper_bound, [n_whales, dimensions])
    
#     (data) = run_woa(
#         initial_positions,
#         functions.rastrigin,
#         lower_bound,
#         upper_bound,
#         dimensions,
#         iterations,
#         n_whales
#     )
#     print("Best fitness value obtained for Sphere function is:", data[-1])
#     print("Best position obtained for Sphere function is:", data[-(dimensions)-1:-1])  
    