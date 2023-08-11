'''
Description: A1Q1d
Author: Noah Kuntner
Date: 2021-01-05 17:24:18
LastEditors: Noah Kuntner
LastEditTime: 2022-01-19 11:39:49
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
        Demand::{str:{str:int}}
            demand of each pair of the cities
        Budget::int
            budget
    '''
    with open(pk, 'rb') as f:
        data = pickle.load(f)

    NbCities = data['NbCities']
    Budget = data['Budget']
    Cities = data['Cities'] 
    CitiesWithin = data['CitiesWithin']
    Cost = data['Cost']
    Demand = data['Demand']

    return NbCities, Budget, Cities, CitiesWithin, Cost, Demand


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
        f.write('Maximum demand fulfilled: {}\n'.format(int(solution.get_objective_value())))
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
    NbCities, Budget, Cities, CitiesWithin, Cost, Demand = load_data(pk)
    
    '''
    print('Number of cities is: ', NbCities)
    print('Budget per City is: \n', Budget)
    print('Total cities are: \n', Cities)
    print('Cities within 1000km for each destination are: \n', CitiesWithin)
    print('Daily operating cost per airport is: \n', Cost)
    print('Total Demand per airport is: \n', Demand)

    '''

    # create model
    m =  Model('a1q1d')

    Paths = [(i,j) for i in Cities for j in Cities]
    
    # Auxiliary Matrix
    Cities_Matrix = {}


    for i, j in Paths:
        if j in CitiesWithin[i]:
            Cities_Matrix[(i,j)] = 1
        else:
            Cities_Matrix[(i,j)] = 0

    # create decision variable
    x = m.binary_var_dict(Cities, name='x')

    # Keeping track of the direct paths, the hubs and the final constrained solution
    DirectPath = m.binary_var_dict(Paths, name='direct_path')
    Hub_Dict = m.binary_var_dict(Paths, name='hub_dict')
    Final_Dict = m.binary_var_dict(Paths, name='final_dict')

    # Considering eligible paths between cities and adjacent hub cities
    for i, j in Paths:
        m.add_constraint(Hub_Dict[i,j] <= x[i])
        m.add_constraint(Hub_Dict[i,j] <= x[j])
        m.add_constraint(Hub_Dict[i,j] >= x[i] + x[j] - 1)


    m.add_constraints(DirectPath[i,j] <= m.sum((x[i] * Cities_Matrix[i, j])+ (x[j] * Cities_Matrix[i, j])+ Hub_Dict[i,j]) for i, j in Paths)

    m.add_constraints(Final_Dict[i,j] <= m.sum(DirectPath[i,j] + m.sum((x[c] * Cities_Matrix[i, c] * Cities_Matrix[c, j]) 
                        + (Hub_Dict[c, j] * Cities_Matrix[i, c])+ (Hub_Dict[i,c] * Cities_Matrix[c, j]) for c in Cities)) for i, j in Paths)

    # Objective Function to Maximize Demand
    m.maximize(m.sum(Demand[i][j]*Final_Dict[i,j] for i,j in Paths))

    # Final Budgeting Constraint
    m.add_constraint(m.sum(x[i] * Cost[i] for i in Cities) <= Budget)

    sol = m.solve(log_output = False)
    m.print_solution()

    # update "Noah Kuntner" to your real name and save the solution
    save_sol(pk, sol, x, Cities, "Noah Kuntner")