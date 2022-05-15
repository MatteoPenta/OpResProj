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
        self.delivery = deliveries
        points.append([0,0]) # depot position
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
        # evaluate the distance matrix
        self.distance_matrix = spatial.distance_matrix(points, points)

        i = 0
        for d in self.delivery:
            # save the original index of the delivery 
            self.delivery[d]['index'] = i
            # evaluate the distance of every delivery from the depot
            self.delivery[d]['dist_from_depot'] = self.distance_matrix[0,i+1]
            # evaluate the score of the delivery
            self.delivery[d]['score'] = self.deliv_crowds_weights['a']*(1-self.delivery[d]['p_failed']) + \
                self.deliv_crowds_weights['b']*self.delivery[d]['dist_from_depot']
            print(f"[DEBUG] Score of node {self.delivery[d]['id']}: {self.delivery[d]['score']}")
            print(f"        Distance of node {self.delivery[d]['id']}: {self.delivery[d]['dist_from_depot']}")
            i += 1

        # 2) evaluate the threshold based on self.quantile
        threshold = np.quantile([d[1]['score'] for d in self.delivery.items()], self.quantile)
        print(f"Threshold: {threshold}")
        threshold_dist = np.quantile([d[1]['dist_from_depot'] for d in self.delivery.items()], self.quantile)
        print(f"Distance threshold: {threshold_dist}")
        
        # 3) select the deliveries with score above threshold
        id_to_crowdship = []
        for _,ele in self.delivery.items():
            if ele['score'] > threshold:
                id_to_crowdship.append(ele['id'])
        
        return id_to_crowdship 

    def compute_VRP(self, delivery_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):
        for d in self.delivery:
            self.delivery[d]['chosen_vrp'] = False

        # sort self.delivery based on their distance from the depot
        sorted(self.delivery.items(), key=lambda x:x[1]['dist_from_depot'])

        sol = []
        for k in range(len(vehicles_dict)):
            # Initialize the solution for the k-th vehicle.
            # Add the depot as the first delivery of the path
            # of the vehicle
            sol.append({'path': [0],'arrival_times':[0], 'waiting_times': [0],
                'vol_left': vehicles_dict[k]['capacity']}) 
            
            # add to the solution of this vehicle the closest delivery that:
            #   - is still available 
            #   - respects the capacity constraint
            #   - corresponds to an arrival time that is lower
            #     than the upper limit of its time window
            aval_d = [d for _,d in self.delivery.items() if (self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and \
                    self.distance_matrix[0,d['index']] < d['time_window_max'])]
            if aval_d:
                sol[k]['path'].append(aval_d[0]['id'])
                sol[k]['arrival_times'].append(self.distance_matrix[0,aval_d[0]['index']])
                sol[k]['waiting_times'].append(max(0,
                    aval_d[0]['time_window_min'] - sol[k]['arrival_times'][-1]))
                sol[k]['vol_left'] -= aval_d[0]['vol']
                aval_d[0]['chosen_vrp'] = True

                # add again the depot, considering the eventual waiting time
                # and the fact that the path is obviously symmetric
                sol[k]['path'].append(0)
                sol[k]['arrival_times'].append(
                    sol[k]['arrival_times'][-1]+sol[k]['waiting_times'][-1]+sol[k]['arrival_times'][-1]
                )

            # the flag will be set to False and then to True again only if 
            # feasible insertions are found for any non-connected node
            feasible_nodes_flag = True
            while(feasible_nodes_flag):
                feasible_nodes_flag = False
                # Consider all deliveries not yet inserted in a solution
                for d in [d for _,d in self.delivery.items() if self.nodeIsFeasibleVRP(d, sol[k]['vol_left'])]:
                    # Find the best position of the delivery d among every
                    # pair of deliveries already in the solution.
                    # Each iteration of this loop considers a different positioning
                    # of the delivery "d", where "prev_n" and "next_n" are the indiced of 
                    # the nodes that precede and follow "d".
                    for i in range(len(sol[k]['path'])-1):
                        prev_n = i
                        next_n = i+1
                        # Check time feasibility considering this insertion

                        
            
        return sol

    def nodeIsFeasibleVRP(self, d, v_vol_left):
        """
        Description
        
        Parameters
            :param          d:  considered delivery
            :type           d:  dictionary
            :param          v_vol_left: volume left in the vehicle
            :type           v_vol_left: integer
        
        Output
            :returns        True/False
            :rtype          bool
        """
        if d['chosen_vrp'] == False and d['vol'] <= v_vol_left:
            return True
        return False

    def nodeTimeFeasibleVRP(self, sol_k, prev_n, u, next_n):
        arr_time_u = prev_n['arrival_time']
        self.distance_matrix[prev_n['index'], next_n['index']] # ...
        # ...


    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
