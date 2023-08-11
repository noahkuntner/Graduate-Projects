'''
Description: A1Q1b
Author: Noah Kuntner
Date: 2021-01-05 17:24:17
LastEditors: Noah Kuntner
LastEditTime: 2022-01-19 11:39:40
'''

import argparse
import itertools
import os
import pickle
from collections import defaultdict
from docplex.mp.model import Model
from pathlib import Path

# load the data
def load_data(pk):
    ''' Load data from pickle
    Args:
        pk::str
            path of the data file
    Returns:
        NbCities::int
            number of the cities
        Cities::[str]
            list of cities
        CitiesWithin::{str:[str]}
            dictionary stores list of cities within 1000km of each city
        Cost::{str:int}
            dictionary stores the cost of each city
    '''
    with open(pk, 'rb') as f:
        data = pickle.load(f)

    NbCities = data['NbCities']
    Cities = data['Cities'] 
    CitiesWithin = data['CitiesWithin']
    Cost = data['Cost']

    return NbCities, Cities, CitiesWithin, Cost


# save solution
def save_sol(pk, solution, x, Cities, name):
    ''' Save the solution of the model to a txt file
    Args:
        pk::str
            path of the data
        solution::docplex.mp.solution.SolveSolution
            holds the results of a solve
        x::binary_var_dict
            decision variable
        Cities::[str]
            list of cities
        name::str
            Your name (last name then first name)
    '''
    with open("{}-{}.txt".format(name, Path(pk).stem), "w") as f:
        f.write('Minimum cost: {}\n'.format(int(solution.get_objective_value())))
        f.write('Hubs are opened at:\n')
        for c in Cities: # or for c in range(NbCities):
            if solution[x[c]]:
                f.write('{}\n'.format(c))


if __name__ == '__main__':

    ### given to the students
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    args = parser.parse_args()

    # load data from pickle
    pk = args.data
    NbCities, Cities, CitiesWithin, Cost = load_data(pk)

    # create model
    m = Model('a1q1b')
    
    # create decision variable
    x = m.binary_var_dict(Cities, lb=0, name='Hubs')

    # Calculate the cost for each city
    Costs_Operation = m.sum(Cost[city] * x[city] for city in Cities)
    m.minimize(Costs_Operation)

    # Add Constraints for cities constraint
    for i in Cities:
        m.add_constraint(m.sum(x[j] for j in CitiesWithin[i]) >= 1, ctname = f'{i}_is_covered')


    sol = m.solve(log_output = False)

    # update "Noah Kuntner" to your real name and save the solution
    save_sol(pk, sol, x, Cities, "Noah Kuntner")