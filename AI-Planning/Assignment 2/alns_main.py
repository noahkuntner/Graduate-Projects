'''
Description: ALNS + EVRP
Author: Noah Kuntner
Date: 2022-02-15 13:38:21
LastEditors: Noah Kuntner
LastEditTime: 2022-02-24 19:43:57
'''
import math
import argparse
import numpy as np
import numpy.random as rnd
import networkx as nx
import matplotlib.pyplot as plt
from lxml import etree as LET
#from sklearn.metrics import euclidean_distances

from evrp import *
from pathlib import Path

import sys
sys.path.append('./ALNS')
from alns import ALNS, State
from alns.criteria import HillClimbing, SimulatedAnnealing, RecordToRecordTravel


def euclidean_distances(A, B):
    '''Calculate Euclidean distance'''
    return math.sqrt((A.y-B.y)**2 + (A.x-B.x)**2)

### draw and output solution ###
def save_output(YourName, evrp, suffix):
    '''Draw the EVRP instance and save the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            'initial' for random initialization
            and 'solution' for the final solution
    '''
    draw_evrp(YourName, evrp, suffix)
    generate_output(YourName, evrp, suffix)

### visualize EVRP ###
def create_graph(evrp):
    '''Create a directional graph from the EVRP instance
    Args:
        evrp::EVRP
            an EVRP object
    Returns:
        g::nx.DiGraph
            a directed graph
    '''
    g = nx.DiGraph(directed=True)
    g.add_node(evrp.depot.id, pos=(evrp.depot.x, evrp.depot.y), type=evrp.depot.type)
    for c in evrp.customers:
        g.add_node(c.id, pos=(c.x, c.y), type=c.type)
    for cs in evrp.CSs:
        g.add_node(cs.id, pos=(cs.x, cs.y), type=cs.type)
    return g

def draw_evrp(YourName, evrp, suffix):
    '''Draw the EVRP instance and the solution
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file, 
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    g = create_graph(evrp)
    route = list(node.id for node in sum(evrp.route, []))
    edges = [(route[i], route[i+1]) for i in range(len(route) - 1) if route[i] != route[i+1]]
    g.add_edges_from(edges)
    colors = []
    for n in g.nodes:
        if g.nodes[n]['type'] == 0:
            colors.append('#0000FF')
        elif g.nodes[n]['type'] == 1:
            colors.append('#FF0000')
        else:
            colors.append('#00FF00')
    pos = nx.get_node_attributes(g, 'pos')
    fig, ax = plt.subplots(figsize=(24, 12))
    nx.draw(g, pos, node_color=colors, with_labels=True, ax=ax, 
            arrows=True, arrowstyle='-|>', arrowsize=12, 
            connectionstyle='arc3, rad = 0.025')

    plt.text(0, 6, YourName, fontsize=12)
    plt.text(0, 3, 'Instance: {}'.format(evrp.name), fontsize=12)
    plt.text(0, 0, 'Objective: {}'.format(evrp.objective()), fontsize=12)
    plt.savefig('{}_{}_{}.jpg'.format(YourName, evrp.name, suffix), dpi=300, bbox_inches='tight')
    
### generate output file for the solution ###
def generate_output(YourName, evrp, suffix):
    '''Generate output file (.txt) for the evrp solution, containing the instance name, the objective value, and the route
    Args:
        YourName::str
            your name, eg. John_Doe
        evrp::EVRP
            an EVRP object
        suffix::str
            suffix of the output file,
            eg. 'initial' for random initialization
            and 'solution' for the final solution
    '''
    str_builder = ['{}\nInstance: {}\nObjective: {}\n'.format(YourName, evrp.name, evrp.objective())]
    for idx, r in enumerate(evrp.route):
        str_builder.append('Route {}:'.format(idx))
        j = 0
        for node in r:
            if node.type == 0:
                str_builder.append('depot {}'.format(node.id))
            elif node.type == 1:
                str_builder.append('customer {}'.format(node.id))
            elif node.type == 2:
                str_builder.append('station {} Charge ({})'.format(node.id, evrp.vehicles[idx].battery_charged[j]))
                j += 1
        str_builder.append('\n')
    with open('{}_{}_{}.txt'.format(YourName, evrp.name, suffix), 'w') as f:
        f.write('\n'.join(str_builder))

### Destroy operators ###
# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_worst_edge(current, random_state):
    
    ''' Destroy operator sample (name of the function is free to change)
    Args:
        current::EVRP
            an EVRP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::EVRP
            the evrp object after destroying
    
    worst_edges = sorted(destroyed.nodes,
                         key=lambda node: distances.euclidean(node[1],
                                                              destroyed.edges[node][1]))

    for idx in range(edges_to_remove(current)):
        del destroyed.edges[worst_edges[-(idx + 1)]]

    '''

    """
    Removes worst edges iteratively, thus the edges that have are the furthest.
    """    

    destroyed = copy.deepcopy(current)
    
    destroyed_tsp = []
    destroyed_tsp.append(destroyed.depot)

    destroyed_tsp.extend([customer for customer in destroyed.customers])
    destroyed_tsp.append(destroyed.depot)
    
    destroyed_distances = [euclidean_distances(destroyed_tsp[i],destroyed_tsp[i+1]) for i in range(len(destroyed.customers)-1)]    
    degree_of_destruction = int(destroyed.destruction*len(destroyed_distances))
    
    idx = np.argpartition(destroyed_distances, -degree_of_destruction)[-degree_of_destruction:]
    
    nodes_destroyed = [destroyed_tsp[i] for i in idx]
    destroyed.customers = [i for i in destroyed.customers if i not in nodes_destroyed]
    
    return destroyed


def destroy_random_edge(current, random_state):
    """ Random removal iteratively removes random edges. """
    
    destroyed = copy.deepcopy(current)      
    degree_of_destruction = int(destroyed.destruction*len(destroyed.customers))
    
    idx = random_state.choice(destroyed.customers,degree_of_destruction,replace=False)
    destroyed.customers = [i for i in destroyed.customers if i not in idx]        
    
    return destroyed


### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def greedy_repair(destroyed, random_state):
    ''' repair operator sample (name of the function is free to change)
    Args:
        destroyed::EVRP
            an EVRP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::EVRP
            the evrp object after repairing
    '''
    # You should code here
    
    nodes_insertion = []
    
    for node in destroyed.customer_visited:
        if node not in destroyed.customers:
            nodes_insertion.append(node)
    
    
    for node in nodes_insertion:
        repaired_node_buildup = destroyed.customers.copy()
        
        node_nearest = node.get_nearest_node(repaired_node_buildup)
        index_nearest = destroyed.customers.index(node_nearest)
        repaired_node_buildup.remove(node_nearest)
        
        node_second_nearest = node.get_nearest_node(repaired_node_buildup)
        index_second_nearest = destroyed.customers.index(node_second_nearest)
        repaired_node_buildup.remove(node_second_nearest)
        
        if index_second_nearest > index_nearest:
            random_idx = random_state.choice(range(index_nearest+1,index_second_nearest+1))
            destroyed.customers.insert(random_idx, node)
            
        elif index_nearest > index_second_nearest:
            random_idx = random_state.choice(range(index_second_nearest+1,index_nearest+1))
            destroyed.customers.insert(random_idx, node)         
    
        
    destroyed.split_route(destroyed.customers)   
    
    print(f"{[i.id for i in destroyed.vehicles[-1].node_visited]}")
    
    return destroyed
      

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    parser.add_argument(dest='seed', type=int, help='seed')
    args = parser.parse_args()
    
    # instance file and random seed
    xml_file = args.data
    seed = int(args.seed)
    
    # load data and random seed
    parsed = Parser(xml_file)
    evrp = EVRP(parsed.name, parsed.depot, parsed.customers, parsed.CSs, parsed.vehicle, destruction = 0.3)
    
    # construct random initialized solution
    evrp.random_initialize(seed)
    print("Solution objective is {}.".format(evrp.objective()))
    
    # visualize initial solution and gernate output file
    save_output('Noah Kuntner', evrp, 'initial')
    
    # ALNS
    random_state = rnd.RandomState(seed)
    alns = ALNS(random_state)
    
    # Add Destroy
    # You should add all your destroy and repair operators
    alns.add_destroy_operator(destroy_worst_edge)
    # Add Destroy
    alns.add_destroy_operator(destroy_random_edge)
    
    # Add Repair
    alns.add_repair_operator(greedy_repair)

    # Run ALNS
    # select cirterion
    criterion = HillClimbing()
    # assigning weights to methods
    omegas = [0.4, 0.3, 0.2, 0.2]
    lambda_ = 0.5
    result = alns.iterate(evrp, omegas, lambda_, criterion,
                          iterations=1000, collect_stats=True)

    # Result
    solution = result.best_state
    objective = solution.objective()
    print('Best heuristic objective is {}.'.format(objective))
    
    # visualize final solution and gernate output file
    save_output('Noah Kuntner', solution, 'solution')
    