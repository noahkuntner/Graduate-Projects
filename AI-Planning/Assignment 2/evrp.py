'''
Description: EVRP State
Author: Noah Kuntner
Date: 2022-03-13 12:27:58 
LastEditors: Noah Kuntner
LastEditTime: 2022-02-24 21:40:59
'''

import math
import copy
from turtle import distance
from tabnanny import check
import numpy as np
import xml.etree.ElementTree as ET
import random

from bisect import bisect_left
from itertools import accumulate
from scipy.spatial.distance import cdist, pdist
from pathlib import Path

import sys
sys.path.append('./ALNS')
from alns import ALNS, State


def euclidean_distances(A, B):
    '''Calculate Euclidean distance'''
    return math.sqrt((A.y-B.y)**2 + (A.x-B.x)**2)


### Parser to parse instance xml file ###
# You should not change this class!
class Parser(object):
    
    def __init__(self, xml_file):
        '''initialize the parser
        Args:
            xml_file::str
                the path to the xml file
        '''
        self.xml_file = xml_file
        self.name = Path(xml_file).stem
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()
        self.depot = None
        self.customers = []
        self.CSs = []
        self.vehicle = None
        self.ini_nodes()
        
    def ini_nodes(self):
        for node in self.root.iter('node'):
            if int(node.attrib['type']) == 0:
                self.depot = Depot(int(node.attrib['id']), int(node.attrib['type']), 
                                   float(node.find('cx').text), float(node.find('cy').text))
            elif int(node.attrib['type']) == 1:
                self.customers.append(Customer(int(node.attrib['id']), int(node.attrib['type']), 
                                               float(node.find('cx').text), float(node.find('cy').text),
                                               None))
            else:
                self.CSs.append(ChargingStation(int(node.attrib['id']), int(node.attrib['type']), 
                                                float(node.find('cx').text), float(node.find('cy').text), 
                                                node.find('custom').find('cs_type').text,
                                                None))
        self.set_customers()
        self.set_CSs()
        self.set_vehicle()
    
    def set_customers(self):
        request = {}
        for r in self.root.iter('request'):
            request[int(r.attrib['id'])] = float(r.find('service_time').text)
        for c in self.customers:
            c.service_time = request[c.id]
    
    def set_CSs(self):
        function = {}
        for f in self.root.iter('function'):
            function[f.attrib['cs_type']] = [(float(b.find('battery_level').text), float(b.find('charging_time').text)) for b in f.findall('breakpoint')]
        for cs in self.CSs:
            cs.breakpoints = function[cs.charging_speed]
    
    def set_vehicle(self):
        for v in self.root.iter('vehicle_profile'):
            self.vehicle = Vehicle(0, self.depot, self.depot,
                                   float(v.find('max_travel_time').text), float(v.find('speed_factor').text), 
                                   float(v.find('custom').find('consumption_rate').text), float(v.find('custom').find('battery_capacity').text))

### Node class ###
# You should not change this class!
class Node(object):
    
    def __init__(self, id, type, x, y):
        '''Initialize a node
        Args:
            id::int
                id of the node
            type::int
                0 for depot, 1 for customer, 2 for charging station
            x::float
                x coordinate of the node
            y::float
                y coordinate of the node
        '''
        self.id = id
        self.type = type
        self.x = x
        self.y = y

    def get_nearest_node(self, nodes):
        '''Find the nearest node in the list of nodes
        Args:
            nodes::[Node]
                a list of nodes
        Returns:
            node::Node
                the nearest node found
        '''
        dis = [cdist([[self.x, self.y]], [[node.x, node.y]], 'euclidean') for node in nodes]
        idx = np.argmin(dis)
        return nodes[idx]
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id and self.type == other.type and self.x == other.x and self.y == other.y
        return False
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}'.format(self.id, self.type, self.x, self.y)

### Depot class ###
# You should not change this class!
class Depot(Node):
    
    def __init__(self, id, type, x, y):
        '''Initialize a depot
        Args:
            id::int
                id of the depot
            type::int
                0 for depot
            x::float
                x coordinate of the depot
            y::float
                y coordinate of the depot
        '''
        super(Depot, self).__init__(id, type, x, y)
        
### Customer class ###
# You should not change this class!
class Customer(Node):
    
    def __init__(self, id, type, x, y, service_time):
        '''Initialize a customer
        Args:
            id::int
                id of the customer
            type::int
                1 for customer
            x::float
                x coordinate of the customer
            y::float
                y coordinate of the customer
            service_time::float
                service time of the customer
        '''
        super(Customer, self).__init__(id, type, x, y)
        self.service_time = service_time
        
    def __str__(self):
        return 'Node id: {}, type: {}, x: {}, y: {}, service_time: {}'.format(self.id, self.type, self.x, self.y, self.service_time)
    
### ChargingStation class ###
# You can add your own helper functions to the class, and not required to use the functions defined
# But you should keep the rest untouched!
class ChargingStation(Node):
    
    def __init__(self, id, type, x, y, charging_speed, breakpoints):
        '''init the charging station
        Args:
            id::int
                id of the charging station
            type::int
                2 for charging station
            x::float
                x coordinate of the charging station
            y::float
                y coordinate of the charging station
            charging_speed::str
                'slow', 'normal', or 'fast'
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        '''
        super(ChargingStation, self).__init__(id, type, x, y)
        self.charging_speed = charging_speed
        self.breakpoints = breakpoints
        
    def charge_to(self, vehicle, x):
        ''' Charge the vehicle to the target battery level and calculate the charging time required
        Args:
            vehicle::Vehicle
                vehicle to be charged
            x::float
                target battery level
        '''
        # Target battery level is larger than the vehicle current battery level
        assert(x > vehicle.battery_level)
        # Target battery level is not larger than the vehicle's battery capacity
        assert(x <= vehicle.battery_capacity)
        
        start_time = ChargingStation.reverse_charging_function(vehicle.battery_level, self.breakpoints)
        end_time = ChargingStation.reverse_charging_function(x, self.breakpoints)
        # the battery charged is recorded
        vehicle.battery_charged.append(x - vehicle.battery_level)
        # the battery level when leaving the CS is recorded
        vehicle.battery_charged_to.append(x)
        # the vehicle's battery level is updated after charging
        vehicle.battery_level = x
        # calculate the charging time required
        vehicle.charging_time += end_time - start_time
        
    def charge_till(self, vehicle, t):
        ''' Charge the vehicle using time t and calculate the battery charged
        Args:
            vehicle::Vehicle
                vehicle to be charged
            t::float
                time to charge 
        '''
        start_time = ChargingStation.reverse_charging_function(vehicle.battery_level, self.breakpoints)
        end_time =  start_time + t
        # calculate the battery level of the vehicle after charging
        battery_level = ChargingStation.charging_function(end_time, self.breakpoints)
        # the battery charged is recorded
        vehicle.battery_charged.append(battery_level - vehicle.battery_level)
        # the battery level when leaving the CS is recorded
        vehicle.battery_charged_to.append(battery_level)
        # the vehicle's battery level is updated after charging
        vehicle.battery_level = battery_level
        # update the charging time for the vehicle
        vehicle.charging_time += t
        
    def __str__(self):
        return 'ChargingStation id: {}, type: {}, x: {}, y: {}, charging_speed: {}, breakpoints: {}'.format(self.id, self.type, self.x, self.y, self.charging_speed, self.breakpoints)
    
    @staticmethod
    def charging_function(t, breakpoints):
        ''' Charging function (x-axis is the time and y-axis is the battery level)
        Give the time t, find the corresponding battery level using the charging function
        Args:
            t::float
                time
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        Returns:
            battery_level::float
                battery level corresponding to time t
        '''
        condlist = [(t > breakpoints[i][1]) & (t <= breakpoints[i + 1][1]) for i in range(len(breakpoints) - 1)]
        k = [(breakpoints[i + 1][0] - breakpoints[i][0])/(breakpoints[i + 1][1] - breakpoints[i][1]) for i in range(len(breakpoints) - 1)]
        funclist = [lambda t: k[0]*t + breakpoints[0][0], 
                    lambda t: breakpoints[1][0] + k[1]*(t - breakpoints[1][1]),
                    lambda t: breakpoints[2][0] + k[2]*(t - breakpoints[2][1])]
        return np.piecewise(float(t), condlist, funclist).reshape(1,)[0]
    
    @staticmethod
    def reverse_charging_function(x, breakpoints):
        ''' Reverse Charging function (x-axis is the battery level and y-axis is the time)
        Give the battery level x, find the corresponding time using the reverse charging function
        Args:
            x::float
                battery level
            breakpoints::[(float, float)]
                a list of breakpoints for the charging function
        Returns:
            time::float
                time corresponding to battery level x
        '''
        condlist = [(x > breakpoints[i][0]) & (x <= breakpoints[i + 1][0]) for i in range(len(breakpoints) - 1)]
        k = [(breakpoints[i + 1][1] - breakpoints[i][1])/(breakpoints[i + 1][0] - breakpoints[i][0]) for i in range(len(breakpoints) - 1)]
        funclist = [lambda x: k[0]*x + breakpoints[0][1], 
                    lambda x: breakpoints[1][1] + k[1]*(x-breakpoints[1][0]),
                    lambda x: breakpoints[2][1] + k[2]*(x - breakpoints[2][0])]
        return np.piecewise(float(x), condlist, funclist).reshape(1,)[0]
    
### Vehicle class ###
# Vehicle class. You could add your own helper functions freely to the class, and not required to use the functions defined
# But please keep the rest untouched!
class Vehicle(object):
    
    def __init__(self, id, start_node, end_node, max_travel_time, speed_factor, consumption_rate, battery_capacity):
        ''' Initialize the vehicle
        Args:
            id::int
                id of the vehicle
            start_node::Node
                starting node of the vehicle
            end_node::Node
                ending node of the vehicle
            max_travel_time::float
                maximum time allowed for the vehicle (including travel and charging time)
            speed_factor::float
                speed factor of the vehicle
            consumption_rate::float
                consumption rate of the vehicle
            battery_capacity::float
                battery capacity of the vehicle
        '''
        self.id = id
        self.start_node = start_node
        self.end_node = end_node
        self.max_travel_time = max_travel_time
        self.speed_factor = speed_factor
        self.consumption_rate = consumption_rate
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity
        # travel time of the vehicle
        self.travel_time = 0
        # charging time of the vehicle
        self.charging_time = 0
        # total battery consumed by the vehicle
        self.battery_consumption = 0
        # all the nodes including depot, customers, or charging stations (if any) visited by the vehicle
        self.node_visited = [self.start_node] # start from depot
        # record the battery charged each time when visiting a charging station
        self.battery_charged = []
        # record the battery level of the vehicle each time when leaving the charging station 
        self.battery_charged_to = []
        
   
    def check_time(self):
        '''Check whether the vehicle's travel time plus charging time is over the maximum travel time or not
        Return True if it is not over the maximum travel time, False otherwise
        '''
        
        if self.travel_time + self.charging_time <= self.max_travel_time:
            return True
        return False
        
    def check_battery(self):
        ''' Check the total battery consumed by the vehicle
        Return True if the battery consumed is not larger than the summation of original battery capacity (when leaving the depot) plus the battery charged, False otherwise
        '''
        if len(self.battery_charged) == 0:
            return self.battery_consumption <= self.battery_capacity
        else:
            return self.battery_consumption <= sum(self.battery_charged) + self.battery_capacity
    
    def check_return(self):
        ''' Check whether the vehicle's return to the depot
        Return True if returned, False otherwise
        '''
        if len(self.node_visited) > 1:
            return self.node_visited[-1] == self.end_node
            
    def __str__(self):
        return 'Vehicle id: {}, start_node: {}, end_node: {}, max_travel_time: {}, speed_factor: {}, consumption_rate: {}, battery_capacity: {}'\
            .format(self.id, self.start_node, self.end_node, self.max_travel_time, self.speed_factor, self.consumption_rate, self.battery_capacity)

### EVRP state class ###
# EVRP state class. You could and should add your own helper functions to the class
# But please keep the rest untouched!
class EVRP(State):
    
    def __init__(self, name, depot, customers, CSs, vehicle, destruction = 0.25):
        '''Initialize the EVRP state
        Args:
            name::str
                name of the instance
            depot::Depot
                depot of the instance
            customers::[Customer]
                customers of the instance
            CSs::[ChargingStation]
                charging stations of the instance
            vehicle::Vehicle
                vehicle of the instance
        '''
        self.name = name
        self.depot = depot
        self.customers = customers
        self.CSs = CSs
        self.vehicle = vehicle
        # record the vehicle used
        self.vehicles = []
        # total travel time of the all the vehicle used
        self.travel_time = 0
        # total charge time of the all the vehicle used
        self.charging_time = 0
        # record the all the customers who have been visited by all the vehicles, eg. [Customer1, Customer2, ..., Customer7, Customer8]
        self.customer_visited = []
        # record the unvisited customers, eg. [Customer9, Customer10]
        self.customer_unvisited = []
        # the route visited by each vehicle, eg. [vehicle1.node_visited, vehicle2.node_visited, ..., vehicleN.node_visited]
        self.route = []
        self.destruction = destruction
                    
    def random_initialize(self, seed=None):
        ''' Randomly initialize the state with split_route() (your construction heuristic)
        Args:
            seed::int
                random seed
        Returns:
            objective::float
                objective value of the state
        '''
        if seed is not None:
            random.seed(606)
        random_tour = copy.deepcopy(self.customers)
        random.shuffle(random_tour)

        self.split_route(random_tour)
        return self.objective()
    
    def copy(self):
        return copy.deepcopy(self)
    
    def split_route(self, tour):
        '''Generate the route given a tour visiting all the customers
        Args:
            tour::[Customer]
                a tour (list of Nodes) visiting all the customers
        Returns:
            num_vehicle
            vehicle_routes

        # You should update the following variables for the EVRP
        EVRP.vehicles
        EVRP.travel_time
        EVRP.charging_time
        EVRP.customer_visited
        EVRP.customer_unvisited
        EVRP.route

        # You should update the following variables for each vehicle used
        Vehicle.travel_time
        Vehicle.charging_time
        Vehicle.battery_consumption
        Vehicle.node_visited
        Vehicle.battery_charged
        Vehicle.battery_charged_to
        '''
        # You should implement your own method to construct the route of EVRP from any tour visiting all the customers
        # Generate distance matrix

        self.vehicles = []
        self.route = []
        self.customers = [i for i in self.customers if i.type!=0] # links to repair method
        self.customer_visited = []
        self.travel_time = 0
        self.charging_time = 0
        tour = [i for i in tour if i.type !=0]
        tour.insert(0, self.depot)
        
        
        self.customer_unvisited = self.customers.copy()
        n = len(tour)
        distance_matrix = [[0 for i in range(n)] for i in range(n)]  # add the depot at first
        cs_distance_matrix = [[0 for j in range(len(self.CSs))] for i in range(n)]

        # Distance matrix for each Customer
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = euclidean_distances(tour[i], tour[j])

        # Charging Station distance matrix to Customer
        for i in range(n):
            for j in range(len(self.CSs)):
                cs_distance_matrix[i][j] = euclidean_distances(tour[i], self.CSs[j])

        def move(current_vehicle: Vehicle, starting_index: int, ending_index: int, recharge: bool = False, charging_station_index: int = None):
            '''Helper function to move Vehicle

            Args:
                current_vehicle::Vehicle
                    Vehicle to udpate

                starting_index::int
                    Starting point. Always a customer or depot

                ending_index::int
                    Ending point. If recharge is True, it is the index of a charging station

                recharge::bool
                    Whether to visit a recharging station in between

                charing_station_index::int
                    Index of charging station to visit. Looks up CSs

            '''

            if recharge:  # Move to the upciming node move through charging station if battery is not full
                # if recharge is true and current battery level is smaller than the capacity: 
                # -> Move to next node through the charging station if battery not full
                
                cs_distance = cs_distance_matrix[starting_index][charging_station_index]
                node_distance = cs_distance_matrix[ending_index][charging_station_index]

                cs_csmpt = cs_distance *current_vehicle.consumption_rate
                cs_time = cs_distance /current_vehicle.speed_factor

                upcoming_node_consumption = node_distance * current_vehicle.consumption_rate
                upcoming_node_time = node_distance / current_vehicle.speed_factor 

                if tour[ending_index].type == 1:
                    service_time = tour[ending_index].service_time
                    
                else:
                    service_time = 0
                
                current_vehicle.node_visited.append(self.CSs[charging_station_index])  # Visit the charging station
                current_vehicle.battery_level -= cs_csmpt  # Subtract the  battery
                current_vehicle.battery_consumption += cs_csmpt 
                self.CSs[selected_charging_station_index].charge_to(current_vehicle, current_vehicle.battery_capacity)  # Charge battery full

                if starting_index != 0:
                    node = tour[starting_index]
                    current_vehicle.node_visited.append(node)  # Visit the new node
                    self.customer_unvisited.remove(node)  # Remove node from unvisited
                    self.customer_visited.append(node)  # Add node to visited
                    
                if ending_index == 0:  # Add the depot node
                    current_vehicle.node_visited.append(self.depot)                    
                current_vehicle.battery_level -= upcoming_node_consumption 
                current_vehicle.battery_consumption += upcoming_node_consumption 
                current_vehicle.travel_time += cs_time + upcoming_node_time + service_time

            else:  # Move to next node
                distance = distance_matrix[starting_index][ending_index]

                if starting_index != 0:
                    node = tour[starting_index]
                    current_vehicle.node_visited.append(node) 
                    self.customer_unvisited.remove(node)  # Remove node from unvisited
                    self.customer_visited.append(node)  # Add node to visited
                    
                if ending_index == 0:  # Add the depot node
                    current_vehicle.node_visited.append(self.depot)
                    
                upcoming_node_consumption = distance * current_vehicle.consumption_rate
                upcoming_node_time = distance / current_vehicle.speed_factor 

                if tour[ending_index].type == 1:
                    service_time = tour[ending_index].service_time
                    
                else:
                    service_time = 0
                
                current_vehicle.battery_level -= upcoming_node_consumption
                current_vehicle.battery_consumption += upcoming_node_consumption
                current_vehicle.travel_time += upcoming_node_time + service_time

        def check_time(current_vehicle: Vehicle, starting_index: int, ending_index: int):
            '''Helper function to check whether moving to next node and immediately returning to depot is time feasible
            Args:
                current_vehicle::Vehicle
                    Vehicle to check

                starting_index::int
                    Starting node index

                ending_index::int
                    Ending point.
            Returns:
                True if feasible, False otherwise
            '''

            upcoming_node_time = distance_matrix[starting_index][ending_index] / current_vehicle.speed_factor
            
            if tour[ending_index].type==1:
                service_time = tour[ending_index].service_time
                
            else:
                service_time = 0
                
            next_node_depot_time = distance_matrix[ending_index][0] / current_vehicle.speed_factor

            return current_vehicle.travel_time + current_vehicle.charging_time + upcoming_node_time + service_time + next_node_depot_time <= current_vehicle.max_travel_time

        def check_battery(current_vehicle: Vehicle, starting_index: int, ending_index: int, recharge=False, return_on_next_node=True):
            '''Function to check whether moving to next node and returning to depot is battery feasible
            Args:
                current_vehicle::Vehicle
                    Vehicle to check

                starting_index::int
                    Starting node index

                ending_index::int
                    Ending node index
            Returns:
                True if battery feasible, False otherwise
            '''

            
            if recharge:
                upcoming_node_consumption = cs_distance_matrix[starting_index][ending_index] * current_vehicle.consumption_rate
                upcoming_node_depot_consumption = cs_distance_matrix[0][ending_index] * current_vehicle.consumption_rate

            else:
                upcoming_node_consumption = distance_matrix[starting_index][ending_index] * current_vehicle.consumption_rate
                upcoming_node_depot_consumption = distance_matrix[ending_index][0] * current_vehicle.consumption_rate

            if return_on_next_node:
                remaining_battery = current_vehicle.battery_level - upcoming_node_consumption - upcoming_node_depot_consumption
                return remaining_battery >= 0

            else:

                remaining_battery = current_vehicle.battery_level - upcoming_node_consumption
                return remaining_battery >= 0

        no_vehicles = 1
        total_routes = []
        self.vehicles.append(copy.deepcopy(self.vehicle))

        # Node is the current node considered, but the future node
        for index, node in enumerate(tour):

            current_vehicle = self.vehicles[no_vehicles-1]
            if len(current_vehicle.node_visited) == 1:
                if check_battery(current_vehicle, 0, index):
                    move(current_vehicle, 0, index)
                else:
                    selected_charging_station_index = np.argmin(cs_distance_matrix[0])
                    move(current_vehicle, 0, index, recharge=True,charging_station_index=selected_charging_station_index)

            if index < n-1:
                
                if check_time(current_vehicle, index, index+1):  # Time check
                    if check_battery(current_vehicle, index, index+1):  # Battery check to next node
                        move(current_vehicle, index, index+1)
                        # move to the upcoming node
                    else:  # If charging is need before moving

                        selected_charging_station_index = np.argmin(cs_distance_matrix[index])  
                        # Checking for closest charging station
                        # Battery feasibility check to nearest charging station
                        if check_battery(current_vehicle, index, selected_charging_station_index, recharge=True):

                            # Move to CS then next node
                            move(current_vehicle, index, index+1, recharge=True,
                                 charging_station_index=selected_charging_station_index)

                        else:  # Battery feasibility check -> move to depot.

                            move(current_vehicle, index, 0)
                            self.route.append(current_vehicle.node_visited)

                            no_vehicles = no_vehicles + 1
                            self.vehicles.append(copy.deepcopy(self.vehicle))  
                            current_vehicle = self.vehicles[no_vehicles-1]

                else:  # Check if it is feasible to move to next node

                    move(current_vehicle, index, 0)
                    self.route.append(current_vehicle.node_visited)

                    no_vehicles = no_vehicles + 1
                    self.vehicles.append(copy.deepcopy(self.vehicle))  
                    current_vehicle = self.vehicles[no_vehicles-1]

            elif check_battery(current_vehicle, index, 0):  # move to the depot
                print(f"Final node for the vehicles: ', {node.id}, ' ', {no_vehicles}")

                move(current_vehicle, index, 0)
                self.route.append(current_vehicle.node_visited)

            else:  # Recharge on the way back to depot
                print(f"Vehicles moving back to recharge before depot: ', {no_vehicles}")
                selected_charging_station_index = np.argmin(
                    cs_distance_matrix[index])
                move(current_vehicle, index, 0, recharge=True,
                     charging_station_index=selected_charging_station_index)
                self.route.append(current_vehicle.node_visited)
        
        print('All unvisited customers', self.customer_unvisited)
       

    
    def objective(self):
        ''' Calculate the objective value of the state
        Return the total travel time and charging time of all vehicles used
        '''
        return self.travel_time +  self.charging_time