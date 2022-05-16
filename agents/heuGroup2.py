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
        # note: alpha1 + alpha2 must be 1 and each of them must be >= 0
        self.alpha1_c1_vrp = 0.5
        self.alpha2_c1_v2p = 0.5

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
                # Consider all deliveries not yet inserted in a solution.
                # For each one, save its best position in the path in "best_pos_all",
                # a dictionary containing the nodes' ids as keys and their best position 
                # in the path as value (stored as a list, see below).
                best_pos_all = {}
                for d in [d for _,d in self.delivery.items() if self.nodeIsFeasibleVRP(d, sol[k]['vol_left'])]:
                    # Find the best position of the delivery d among every
                    # pair of deliveries already in the solution.
                    # Each iteration of this loop considers a different positioning
                    # of the delivery "d", where "prev_n" and "next_n" are the indiced of 
                    # the nodes that precede and follow "d".
                    
                    # Initialize the list containing the best positioning for "d" in terms
                    # of previous / following nodes and cost function c1.
                    #   best_pos_d = [<prev_n>, <next_n>, <c1>]
                    # where <prev_n> and <next_n> are given as indices in the sol[k] lists.
                    best_pos_d = ['','',0]
                    for i in range(len(sol[k]['path'])-1):
                        # Indexes of the nodes that precede and follow "d" inside
                        # the lists contained in sol[k]
                        prev_n = i
                        next_n = i+1
                        # Check time feasibility considering this insertion
                        if self.nodeTimeFeasibleVRP(sol[k], prev_n, d, next_n):
                            feasible_nodes_flag = True
                            # If this positioning of "d" is feasible, evaluate its
                            # cost c1 and compare it with the minimum found
                            c1 = self.getC1(sol[k], prev_n, d, next_n)
                            if not best_pos_d[0]: # first time
                                best_pos_d = [prev_n, next_n, c1]
                            elif c1 < best_pos_d[2]: # found a better placing
                                # update the min
                                best_pos_d = [prev_n, next_n, c1]
                    if best_pos_d[0]: # if a best placing was found, add it to "best_pos_all"
                        best_pos_all[d['id']] = best_pos_d

                # Choose which one of the nodes d (for which a best placing inside this path was found)
                # will be included in the path of the k-th vehicle. 
                # The choice is based on the cost function C2.
                if feasible_nodes_flag:
                    best_d_id = self.getBestDelivery()
                    # 1) Add the new delivery before the depot at the end of the path
                    # 2) insert the new arrival and waiting times
                    # 3) update all the arrival & waiting times of the following deliveries
                    # 4) update the volume left in the vehicle
                    self.updatePath(sol[k], best_d_id, best_pos_all)
                    
                    # 5) set the chosen_vrp of the delivery to True
                    self.delivery[best_d_id]['chosen_vrp'] = True

        return [s['path'] for s in sol]

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

    def nodeTimeFeasibleVRP(self, sol_k, prev_n_sol, d, next_n_sol):
        """
        Description

        Parameters
            :param
            :type
            :param
            :type
            :param
            :type
            :param
            :type

        Output
            :returns        True/False
            :rtype          bool
        """
        # check if the arrival time of d is lower than the upper bound of its
        # time window
        prev_n_id = sol_k['path'][prev_n_sol]
        prev_n_index = self.delivery[prev_n_id]['index'] # index of prev_n in the distance matrix
        dist_prev_d = self.distance_matrix[prev_n_index, d['index']] # distance between prev_n and d
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
            sol_k['waiting_times'][prev_n_sol] + \
            dist_prev_d
        if arr_time_d > d['time_window_max']:
            return False
        
        # Evaluate the first Push-forward PF
        next_n_id = sol_k['path'][next_n_sol]
        next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
        dist_d_next = self.distance_matrix[d['index'], next_n_index] # distance between d and next_n
        waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
        next_n_timeupperbound = self.delivery[next_n_id]['time_window_max']
        PF = (arr_time_d + waiting_time_d + dist_d_next) - sol_k['arrival_times'][next_n_sol]
        for next_n_sol in range(next_n_sol, len(sol_k['path'])-1):
            # If PF == 0: time feasibility is guaranteed from this point on. Return true
            # If PF + arrival time exceeds the time window upper bound, return False
            if PF == 0:
                return True
            elif sol_k['arrival_times'][next_n_sol] + PF > next_n_timeupperbound:
                return False
            
            # If it hasn't returned, update PF to check the next delivery in the path.
            # NOTE: if the node after next_n_sol in the path is the depot, stop 
            if next_n_sol + 1 != len(sol_k['path']):
                PF = max(0, PF - sol_k['waiting_times'][next_n_sol+1])

        # if it has arrived to this point, it means that the time feasibility is 
        # respected for all deliveries in the path after the new one
        return True


    def getC1(self, sol_k, prev_n_sol, d, next_n_sol):
        """
        Description

        Parameters
            :param
            :type
            :param
            :type
            :param
            :type
            :param
            :type

        Output
            :returns        True/False
            :rtype          bool
        """
        prev_n_id = sol_k['path'][prev_n_sol]
        prev_n_index = self.delivery[prev_n_id]['index'] # index of prev_n in the distance matrix
        dist_prev_d = self.distance_matrix[prev_n_index, d['index']] # distance between prev_n and d

        next_n_id = sol_k['path'][next_n_sol]
        next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
        dist_d_next = self.distance_matrix[d['index'], next_n_index] # distance between d and next_n

        dist_prev_next = self.distance_matrix[prev_n_index, next_n_index]

        c11 = dist_prev_d + dist_d_next - dist_prev_next
        
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
                sol_k['waiting_times'][prev_n_sol] + \
                dist_prev_d
        waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        c12 = new_arr_time_next - sol_k['arrival_times'][next_n_sol]

        c1 = self.alpha1_c1_vrp*c11 + self.alpha2_c1_v2p*c12
        return c1

    def getC2(self):
        """
        """

    def getBestDelivery(self):
        """
        """

    def updatePath(self, sol_k, best_d_id, best_pos_all):
        """
        """
        prev_n_sol = best_pos_all[best_d_id][0]
        next_n_sol = best_pos_all[best_d_id][1]
        # 1) Add the new delivery in the chosen place
        sol_k['path'].insert(prev_n_sol+1, best_d_id)
        # update the index of next_n
        next_n_sol += 1

        # 2) insert the new arrival and waiting times
        prev_n_id = sol_k['path'][prev_n_sol]
        prev_n_index = self.delivery[prev_n_id]['index'] # index of prev_n in the distance matrix
        dist_prev_d = self.distance_matrix[prev_n_index, self.delivery[best_d_id]['index']] # distance between prev_n and d
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
            sol_k['waiting_times'][prev_n_sol] + \
                dist_prev_d
        waiting_time_d = max(0, self.delivery[best_d_id]['time_window_min'] - arr_time_d)
        sol_k['arrival_times'].insert(prev_n_sol+1, arr_time_d)
        sol_k['waiting_times'].insert(prev_n_sol+1, waiting_time_d)


        # 3) update all the arrival & waiting times of the following deliveries
        next_n_id = sol_k['path'][next_n_sol]
        next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
        dist_d_next = self.distance_matrix[d['index'], next_n_index] # distance between d and next_n
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        additional_delay_flag = True
        while additional_delay_flag and next_n_sol < len(sol_k['path'])-1:
            # Update the arrival time at next_n
            sol_k['arrival_times'][next_n_sol] = new_arr_time_next
            arr_time_relative = self.delivery[next_n_id]['time_window_min']-new_arr_time_next
            sol_k['waiting_times'][next_n_sol] = max(0, arr_time_relative)
            if arr_time_relative < 0: # if the arrival at next_n is after the lower bound of its time window
                # update the arrival time at the delivery after "next_n" taking into consideration
                # the delay that was introduced at "next_n"
                next_n_sol += 1
                if next_n_sol < len(sol_k['path'])-1:
                    new_arr_time_next = sol_k['arrival_times'][next_n_sol] - arr_time_relative
            else: # otherwise, no additional delay has been introduced from next_n on in the path: exit the while loop 
                additional_delay_flag = False
        
        # 4) update the volume left in the vehicle
        sol_k['vol_left'] -= self.delivery[best_d_id]['vol']


    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass