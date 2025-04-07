import numpy as np
import random
import matplotlib.pyplot as plt

# Problem dimension.
dim = 5

# Test problem evaluation: computes the two objectives.
def evaluate(individual):
    x = individual['x']
    f1 = np.sum(x**2)
    f2 = np.sum((x - 1)**2)
    return np.array([f1, f2])

# Gaussian mutation: adds Gaussian noise to each decision variable.
def gaussian_mutation(individual, sigma=0.1, lower_bound=-10, upper_bound=10):
    mutant = individual.copy()
    mutant['x'] = mutant['x'] + np.random.normal(0, sigma, size=mutant['x'].shape)
    mutant['x'] = np.clip(mutant['x'], lower_bound, upper_bound)
    mutant['objectives'] = evaluate(mutant)
    return mutant

# Tournament selection (selects randomly between two candidates).
def tournament_selection(population):
    i, j = random.sample(range(len(population)), 2)
    return population[i] if random.random() < 0.5 else population[j]

# Standard non-dominated sorting for 2 objectives.
def non_dominated_sort(population):
    remaining = population.copy()
    fronts = []
    while remaining:
        front = []
        for ind in remaining:
            dominated = False
            for other in remaining:
                if id(other) != id(ind):
                    if np.all(other['objectives'] <= ind['objectives']) and np.any(other['objectives'] < ind['objectives']):
                        dominated = True
                        break
            if not dominated:
                front.append(ind)
        fronts.append(front)
        front_ids = {id(ind) for ind in front}
        remaining = [ind for ind in remaining if id(ind) not in front_ids]
    return fronts

# Compute the 2-D hypervolume for a set of minimization points given a reference point.
def compute_hv_2d(points, ref):
    if len(points) == 0:
        return 0.0
    sorted_points = points[np.argsort(points[:, 0])]
    hv = 0.0
    f2_prev = ref[1]
    for p in sorted_points:
        hv += (ref[0] - p[0]) * (f2_prev - p[1])
        f2_prev = p[1]
    return hv

# SMS-EMOA implementation (steady-state).
def sms_emoa(pop_size=100, iterations=5000, sigma=0.1, lower_bound=-10, upper_bound=10):
    population = []
    for _ in range(pop_size):
        ind = {}
        ind['x'] = np.random.uniform(lower_bound, upper_bound, dim)
        ind['objectives'] = evaluate(ind)
        population.append(ind)
    
    # Save initial population for plotting.
    initial_population = [ind.copy() for ind in population]
    
    for it in range(iterations):
        parent = tournament_selection(population)
        offspring = gaussian_mutation(parent, sigma, lower_bound, upper_bound)
        population.append(offspring)
        
        all_points = np.array([ind['objectives'] for ind in population])
        ref = [all_points[:, 0].max(), all_points[:, 1].max()]
        
        fronts = non_dominated_sort(population)
        worst_front = fronts[-1]
        
        worst_points = np.array([ind['objectives'] for ind in worst_front])
        hv_total = compute_hv_2d(worst_points, ref)
        
        contributions = []
        for i in range(len(worst_points)):
            subset = np.delete(worst_points, i, axis=0)
            hv_without = compute_hv_2d(subset, ref)
            contributions.append(hv_total - hv_without)
        
        idx_worst = np.argmin(contributions)
        worst_candidate = worst_front[idx_worst]
        
        # Remove worst_candidate by comparing its id.
        for i, ind in enumerate(population):
            if id(ind) == id(worst_candidate):
                del population[i]
                break
        
        if (it + 1) % 500 == 0:
            print(f"Iteration {it+1}/{iterations} completed.")
    
    return initial_population, population

if __name__ == "__main__":
    init_pop, final_pop = sms_emoa(pop_size=100, iterations=5000, sigma=0.1)
    
    init_objs = np.array([ind['objectives'] for ind in init_pop])
    final_objs = np.array([ind['objectives'] for ind in final_pop])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(init_objs[:, 0], init_objs[:, 1], color='red', label='Initial Population')
    plt.scatter(final_objs[:, 0], final_objs[:, 1], color='blue', label='Final Population')
    plt.xlabel('f1(x) = sum(x_i^2)')
    plt.ylabel('f2(x) = sum((x_i-1)^2)')
    plt.title('Objective Space: Initial (red) vs. Final (blue) Populations (SMS-EMOA)')
    plt.legend()
    plt.grid(True)
    plt.savefig("SMS_EMOA_scatter.png", dpi=300)
    plt.close()
    print("Overall scatter plot saved as SMS_EMOA_scatter.png")
    
    fronts = non_dominated_sort(final_pop)
    if fronts:
        pareto_pop = fronts[0]
        pareto_objs = np.array([ind['objectives'] for ind in pareto_pop])
        margin = 0.1
        f1_min, f1_max = pareto_objs[:, 0].min(), pareto_objs[:, 0].max()
        f2_min, f2_max = pareto_objs[:, 1].min(), pareto_objs[:, 1].max()
        f1_range = f1_max - f1_min if f1_max - f1_min > 0 else 1
        f2_range = f2_max - f2_min if f2_max - f2_min > 0 else 1

        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_objs[:, 0], pareto_objs[:, 1], color='blue', label='Pareto Front')
        plt.xlabel('f1(x) = sum(x_i^2)')
        plt.ylabel('f2(x) = sum((x_i-1)^2)')
        plt.title('Zoomed-in Pareto Front (SMS-EMOA)')
        plt.legend()
        plt.grid(True)
        plt.xlim(f1_min - margin * f1_range, f1_max + margin * f1_range)
        plt.ylim(f2_min - margin * f2_range, f2_max + margin * f2_range)
        plt.savefig("SMS_EMOA_pareto_zoom.png", dpi=300)
        plt.close()
        print("Zoomed Pareto front plot saved as SMS_EMOA_pareto_zoom.png")
    else:
        print("No Pareto front found in the final population.")
