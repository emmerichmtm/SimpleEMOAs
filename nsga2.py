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

# Binary tournament selection based on rank and crowding distance.
def tournament_selection(population):
    i, j = random.sample(range(len(population)), 2)
    p1, p2 = population[i], population[j]
    if p1['rank'] < p2['rank']:
        return p1
    elif p1['rank'] > p2['rank']:
        return p2
    else:
        return p1 if p1['distance'] > p2['distance'] else p2

# Check if vector a dominates vector b.
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# Non-dominated sorting: returns a list of fronts (each front is a list of indices).
def non_dominated_sort(population):
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    rank = [0 for _ in range(len(population))]
    fronts = [[]]
    
    for p in range(len(population)):
        S[p] = []
        n[p] = 0
        for q in range(len(population)):
            if dominates(population[p]['objectives'], population[q]['objectives']):
                S[p].append(q)
            elif dominates(population[q]['objectives'], population[p]['objectives']):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 1
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    if not fronts[-1]:
        fronts.pop()
    
    for i, front in enumerate(fronts):
        for idx in front:
            population[idx]['rank'] = i + 1
    return fronts

# Crowding distance assignment for individuals in a front.
def crowding_distance(population, front):
    l = len(front)
    if l == 0:
        return
    for idx in front:
        population[idx]['distance'] = 0
    
    num_objectives = len(population[front[0]]['objectives'])
    
    for m in range(num_objectives):
        front_objs = [population[idx]['objectives'][m] for idx in front]
        sorted_idx = np.argsort(front_objs)
        f_max = front_objs[sorted_idx[-1]]
        f_min = front_objs[sorted_idx[0]]
        
        # Assign infinite distance to boundary points.
        population[front[sorted_idx[0]]]['distance'] = float('inf')
        population[front[sorted_idx[-1]]]['distance'] = float('inf')
        
        if f_max == f_min:
            continue
        
        for i in range(1, l - 1):
            next_obj = population[front[sorted_idx[i + 1]]]['objectives'][m]
            prev_obj = population[front[sorted_idx[i - 1]]]['objectives'][m]
            population[front[sorted_idx[i]]]['distance'] += (next_obj - prev_obj) / (f_max - f_min)

# Create offspring population using tournament selection and Gaussian mutation.
def create_offspring(population, offspring_size, sigma=0.1, lower_bound=-10, upper_bound=10):
    offspring = []
    while len(offspring) < offspring_size:
        parent = tournament_selection(population)
        child = gaussian_mutation(parent, sigma, lower_bound, upper_bound)
        offspring.append(child)
    return offspring

# NSGA-II main algorithm.
def nsga2(pop_size=100, generations=50, sigma=0.1, lower_bound=-10, upper_bound=10):
    # Initialize population randomly.
    population = []
    for _ in range(pop_size):
        ind = {}
        ind['x'] = np.random.uniform(lower_bound, upper_bound, dim)
        ind['objectives'] = evaluate(ind)
        ind['rank'] = None
        ind['distance'] = None
        population.append(ind)
    
    # Perform initial non-dominated sorting and crowding distance assignment.
    fronts = non_dominated_sort(population)
    for front in fronts:
        crowding_distance(population, front)
    
    # Save a copy of the initial population.
    initial_population = [ind.copy() for ind in population]
    
    for gen in range(generations):
        offspring = create_offspring(population, pop_size, sigma, lower_bound, upper_bound)
        combined_population = population + offspring
        fronts = non_dominated_sort(combined_population)
        for front in fronts:
            crowding_distance(combined_population, front)
        
        new_population = []
        i = 0
        while len(new_population) + len(fronts[i]) <= pop_size:
            for idx in fronts[i]:
                new_population.append(combined_population[idx])
            i += 1
        
        remaining = pop_size - len(new_population)
        if remaining > 0:
            last_front = [combined_population[idx] for idx in fronts[i]]
            last_front.sort(key=lambda ind: ind['distance'], reverse=True)
            new_population.extend(last_front[:remaining])
        
        population = new_population
        best_front = [combined_population[idx] for idx in fronts[0]]
        print(f"Generation {gen + 1}: Best front size = {len(best_front)}")
    
    return initial_population, population

# Run NSGA-II with the test problem and obtain initial and final populations.
if __name__ == "__main__":
    init_pop, final_pop = nsga2(pop_size=100, generations=50, sigma=0.1)
    
    # Extract objective values for initial and final populations.
    init_objs = np.array([ind['objectives'] for ind in init_pop])
    final_objs = np.array([ind['objectives'] for ind in final_pop])
    
    # First plot: overall objective space scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(init_objs[:, 0], init_objs[:, 1], color='red', label='Initial Population')
    plt.scatter(final_objs[:, 0], final_objs[:, 1], color='blue', label='Final Population')
    plt.xlabel('f1(x) = sum(x_i^2)')
    plt.ylabel('f2(x) = sum((x_i-1)^2)')
    plt.title('Objective Space: Initial (red) and Final (blue) Populations')
    plt.legend()
    plt.grid(True)
    plt.savefig("NSGA2_scatter.png", dpi=300)
    plt.close()
    print("Overall scatter plot saved as NSGA2_scatter.png")
    
    # Second plot: zoom in on the Pareto front (rank 1 individuals).
    # Extract Pareto front from final population.
    pareto_pop = [ind for ind in final_pop if ind['rank'] == 1]
    pareto_objs = np.array([ind['objectives'] for ind in pareto_pop])
    
    if len(pareto_objs) > 0:
        # Determine plot limits with some margin.
        margin = 0.1  # 10% margin
        f1_min, f1_max = pareto_objs[:, 0].min(), pareto_objs[:, 0].max()
        f2_min, f2_max = pareto_objs[:, 1].min(), pareto_objs[:, 1].max()
        f1_range = f1_max - f1_min if f1_max - f1_min > 0 else 1
        f2_range = f2_max - f2_min if f2_max - f2_min > 0 else 1
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_objs[:, 0], pareto_objs[:, 1], color='blue', label='Pareto Front')
        plt.xlabel('f1(x) = sum(x_i^2)')
        plt.ylabel('f2(x) = sum((x_i-1)^2)')
        plt.title('Zoomed-in Pareto Front')
        plt.legend()
        plt.grid(True)
        plt.xlim(f1_min - margin * f1_range, f1_max + margin * f1_range)
        plt.ylim(f2_min - margin * f2_range, f2_max + margin * f2_range)
        plt.savefig("NSGA2_pareto_zoom.png", dpi=300)
        plt.close()
        print("Zoomed Pareto front plot saved as NSGA2_pareto_zoom.png")
    else:
        print("No Pareto front (rank 1 individuals) found in the final population.")
