# -*- coding: utf-8 -*-
import time
from turtle import distance
from agents import *
from scipy import spatial
import numpy as np

class HeuGroup2(Agent):

    def __init__(self, env):
        self.name = "HeuGroup2"
        self.env = env
        self.quantile = 0.5
        self.deliv_crowds_weights = {
            "a": 0.5,
            "b": 0.5
        }

    def compute_delivery_to_crowdship(self, deliveries):
        # 1) evaluate the score for all deliveries
        if len(deliveries) == 0:
            return []
        points = []
        self.delivery = []
        points.append([0,0]) # depot position
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        # evaluate the distance matrix
        self.distance_matrix = spatial.distance_matrix(points, points)

        for i in range(len(self.delivery)):
            # evaluate the distance of every delivery from the depot
            self.delivery[i]['dist_from_depot'] = self.distance_matrix[0,i+1]
            # evaluate the score of the delivery
            self.delivery[i]['score'] = self.deliv_crowds_weights['a']*(1-self.delivery[i]['p_failed']) + \
                self.deliv_crowds_weights['b']*self.delivery[i]['dist_from_depot']
            print(f"[DEBUG] Score of node {self.delivery[i]['id']}: {self.delivery[i]['score']}")
            print(f"        Distance of node {self.delivery[i]['id']}: {self.delivery[i]['dist_from_depot']}")

        # 2) evaluate the threshold based on self.quantile
        threshold = np.quantile([dlv['score'] for dlv in self.delivery], self.quantile)
        print(f"Threshold: {threshold}")
        threshold_dist = np.quantile([d['dist_from_depot'] for d in self.delivery], self.quantile)
        print(f"Distance threshold: {threshold_dist}")
        
        # 3) select the deliveries with score above threshold
        id_to_crowdship = []
        for ele in self.delivery:
            if ele['score'] > threshold:
                id_to_crowdship.append(ele['id'])
        
        return id_to_crowdship 

    def compute_VRP(self, delivery_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):
        for i in range(len(self.delivery)):
            self.delivery[i]['chosen_vrp'] = False

        # sort self.delivery based on their distance from the depot
        self.delivery.sort(key=lambda x:x['dist_from_depot'])

        sol = []
        for k in range(len(vehicles_dict)):
            sol[k] = [] # initialize the solution for the k-th vehicle
            sol[k].append(0) # add the depot to the k-th solution
            
            # add the closest delivery that is still available to the 
            # solution of this vehicle
            aval_d = [d for d in self.delivery if d['chosen_vrp'] == False]
            if aval_d:
                sol[k].append(aval_d[0])
                aval_d[0]['chosen_vrp'] = True

            # the flag will be set to False and then to True again only if 
            # feasible insertions are found for any non-connected node
            feasible_nodes_flag = True
            while(feasible_nodes_flag):
                feasible_nodes_flag = False
                # Consider all deliveries not yet inserted in a solution
                for d in [d for d in self.delivery if d['chosen_vrp'] == False]:
                    # Find the best position of the delivery d among every
                    # pair of deliveries already in the solution
                    print("CONTINUA DA QUI")
            
        return sol

    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
