import random

from deap import algorithms, base, creator, tools
import numpy as np

from flux_center import (
    center_flux,
    center_flux_normalized,
    two_points_flux_normalized,
    path_flux_normalized,
    path_v2_flux_normalized,
    path_v3_flux_normalized,
    path_v4_flux_normalized,
)


def evaluate(individual):
    # return center_flux(np.array([individual])),
    # return center_flux_normalized(np.array(individual)),
    return (two_points_flux_normalized(np.array(individual)),)
    # return path_flux_normalized(np.array(individual)),
    # return path_v2_flux_normalized(np.array(individual)),
    # return path_v3_flux_normalized(np.array(individual)),


def main():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, 1)
    # Structure initializers
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 25
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(6)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=30,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )
    print(hof)


if __name__ == "__main__":
    main()
