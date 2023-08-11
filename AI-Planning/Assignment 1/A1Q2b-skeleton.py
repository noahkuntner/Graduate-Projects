import os
import pickle
import argparse
import pandas as pd
from collections import defaultdict
from docplex.cp.model import CpoModel
from pathlib import Path


# load the data
def load_data(pk):
    ''' Load data from pickle
    Args:
        pk::str
            path of the data file
    Returns:
        NbSlots::int
            number of slots (slot id starts from 0)
        NbRooms::int
            number of rooms (room id starts from 0)
        NbProfs::int
            number of professors
        NbCourses::int
            number of courses
        EligibleProfs::{int: {int}}
            dictionary of set of professors who are eligible to teach the course
            the key is the course id (start from 0)
            the value is a set of professor ids (start from 0) who are eligible to teach the specific course
        NbAllocatedCourses::{int: {int}}
            dictionary of teaching load, i.e. the number of courses per week that the professor has been allocated to
            the key is the professor id (start from 0)
            the value is the number of courses per week that he/ she has been allocated to
    '''
    with open(pk, 'rb') as f:
        dat = pickle.load(f)

    NbSlots = dat['NbSlots']
    NbRooms = dat['NbRooms']
    NbProfs = dat['NbProfs']
    NbCourses = dat['NbCourses']
    EligibleProfs = dat['EligibleProfs']
    NbAllocatedCourses = dat['NbAllocatedCourses']

    return NbSlots, NbRooms, NbProfs, NbCourses, EligibleProfs, NbAllocatedCourses


# create schedule from the solution
def create_schedule(sol, t, r, p):
    schedule = defaultdict(list)
    for i in range(len(t)):
        schedule[i].append(sol[t[i]])
    for j in range(len(r)):
        schedule[j].append(sol[r[j]])
    for k in range(len(p)):
        schedule[k].append(sol[p[k]])
    return schedule


# transform schedule to a readable string
def print_schedule(sol, t, r, p):
    Rooms = ["SR2-1", "SR2-2", "SR2-3", "SR2-4"]

    TimeSlots = ["Mon1", "Mon2", "Mon3", "Tue1", "Tue2", "Tue3" ,"Wed1", "Wed2", "Wed3",
             "Thu1", "Thu2", "Thu3", "Fri1", "Fri2", "Fri3"]

    Profs =  ["Amy", "Bob" ,"Cindy", "David", "Eva", "Faye", "Grey", "Harry", "Iva", "Jack",
            "Andy", "Bobby" ,"Charles", "Daisy", "Emily", 
            "Ava", "Brayden" ,"Chloe", "Daniel", "Ellie"]

    Courses = ["CS601", "CS602", "CS603", "CS604", "CS605" ,"CS606", "CS607", "CS608", "CS609", "CS610",
           "IS701", "IS702", "IS703", "IS704", "IS705" ,"IS706", "IS707", "IS708", "IS709", "IS710",
           "IS421", "IS422", "IS423", "IS424", "IS425" ,"IS426", "IS427", "IS428", "IS429", "IS430",
           "IS521", "IS522", "IS523", "IS524", "IS525" ,"IS526", "IS527", "IS528", "IS529", "IS530",
           "IS621", "IS622", "IS623", "IS624", "IS625" ,"IS626", "IS627", "IS628", "IS629", "IS630",
           "CS401", "CS402", "CS403", "CS404", "CS405" ,"CS406", "CS407", "CS408", "CS409", "CS410"]
           
    schedule = create_schedule(sol, t, r, p)
    if None not in schedule[0]:
        return '\n'.join([
        'Course {} is taught by {} in room {} at timeslot {}'.format(
            Courses[c], Profs[schedule[c][2]], Rooms[schedule[c][1]], TimeSlots[schedule[c][0]]) for c in sorted(schedule.keys())
    ])
    else:
        return "There is no feasible solution"


# visualize schedule as a dataframe
def visualize_schedule(NbSlots, NbRooms, sol, t, r):
    schedule = create_schedule(sol, t, r)
    data = [[''] * NbSlots for _ in range(NbRooms)]
    for i in schedule:
        data[schedule[i][1]][schedule[i][0]] = 'Course {}'.format(i)
    df = pd.DataFrame(data)
    df.columns = ['Slot {}'.format(i) for i in range(NbSlots)]
    df.index = ['Room {}'.format(i) for i in range(NbRooms)]
    return df


# save solution (given to the students)
def save_sol(pk, sol, t, r, p, name):
    with open("{}-{}.txt".format(name, Path(pk).stem), "w") as text_file:
        text_file.write(print_schedule(sol, t, r, p))


if __name__ == '__main__':

    ### given to the students
    parser = argparse.ArgumentParser(description='load data')
    parser.add_argument(dest='data', type=str, help='data')
    args = parser.parse_args()

    # load data from pickle
    pk = args.data
    NbSlots, NbRooms, NbProfs, NbCourses, EligibleProfs, NbAllocatedCourses = load_data(
        pk)

    ### create base model
    m = CpoModel(name="TimeTable")
    
    # build your model here
    print('Eligible Slots are: ', NbSlots)
    print('\nNumber of Rooms are: ', NbRooms)
    print('\nNumber of Profs are: ', NbProfs)
    print('\nNumber of Courses are: ', NbCourses)
    print('\nEligible Profs are: ', EligibleProfs)
    print('\nMaximum teachable courses are: ', NbAllocatedCourses)
    
    # Decision Variables
    # Professor # Room # Time
    t = m.integer_var_list(NbCourses, 0, NbSlots - 1, "T")
    r = m.integer_var_list(NbCourses, 0, NbRooms - 1, "R")
    p = m.integer_var_list(NbCourses, 0, NbProfs - 1, "P")


    for x in range(NbCourses):
        for i in range(NbProfs):

            # Only consider professors that are eligible
            m.add(m.if_then(p[x] == i, i in EligibleProfs[x]))
            
            # Constraints;  
         
        for y in range(NbCourses):
            if x != y:
                # Each professor can only teach one module every day
                #print('x is equal to ', x)
                #print('y is equal to ', y)
                m.add(m.if_then(p[x] == p[y], t[x] // 3 != t[y] // 3))
                m.add(m.if_then(r[x] == r[y], t[x] != t[y]))
               
    
    for i in range(NbProfs):
        # Each prof can perform at most perform their maximum courses and one less at most 
        m.add(m.count(p, i) <= NbAllocatedCourses[i])
        m.add(m.count(p, i) >= NbAllocatedCourses[i] - 1)
    
    # Objective Function

    sol = m.solve(log_output=True)

    # update "Noah Kuntner" to your real name and save the solution
    save_sol(pk, sol, t, r, p, "Noah Kuntner")