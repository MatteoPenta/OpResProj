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
        self.quantile = 1 #DEBUG
        self.deliv_crowds_weights = {
            "a": 0.5,
            "b": 0.5
        }
        # note: alpha1 + alpha2 must be 1 and each of them must be >= 0
        self.mu_vrp = 1
        self.alpha1_c1_vrp = 0.8
        self.alpha2_c1_v2p = 0.2
        self.lambda_vrp = 1
        self.volw = 1 # weight associated to the volume of the delivery

        # ALNS Parameters
        self.alns_N_max = 100 # max number of iterations
        self.alns_N_IwI = 30 # max number of iterations without an improvement
        self.alns_N_s = 5 # number of iterations in a segment
        self.alns_mu = 0.05 # tuning parameter for the "temperature" of a solution
        self.alns_eps = 0.9998  # cooling rate for the temperature
        # parameters used to increment the scores of the operators
        self.alns_sigma1 = 20 # if the new sol is better than the best one
        self.alns_sigma2 = 16 # if the new sol is better than the curr one
        self.alns_sigma3 = 13 # if the new sol is NOT better than the curr one but it is chosen
        self.alns_rho = 0.1 # "reaction factor" used to update the weights of the operators

        # Repair algorithms (ALNS)
        #   'p': probability of the algorithm
        #   'w': weight of the algorithm
        #   's': score of the algorithm in the current segment (used to evaluate the weight)
        #   'n': number of times the algorithm was chosen in the current segment
        self.repair_algos = {
            'greedy': {'func': self.alns_repair_greedy, 'p': 0.25, 'w': 1, 's': 0, 'n': 0},
            'rand_vehicle': {'func': self.alns_repair_rand_vehicle, 'p': 0.25, 'w': 1, 's': 0, 'n': 0},
            'rand_choice': {'func': self.alns_repair_rand_choice, 'p': 0.25, 'w': 1, 's': 0, 'n': 0},
            'closest_pair': {'func': self.alns_repair_closest_pair, 'p': 0.25, 'w': 1, 's': 0, 'n': 0}
        }

        # Destroy algorithms (ALNS)
        self.destroy_algos = {
            
        }
    
    # ALNS Heuristics
    def alns_repair_greedy(self, arg):
        print("Greedy ALNS")

    def alns_repair_rand_vehicle(self, arg):
        print("Greedy ALNS")

    def alns_repair_rand_choice(self, arg):
        print("Greedy ALNS")

    def alns_repair_closest_pair(self, arg):
        print("Greedy ALNS")

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

        i = 1
        for d in self.delivery:
            # save the original index of the delivery 
            self.delivery[d]['index'] = i
            # evaluate the distance of every delivery from the depot
            self.delivery[d]['dist_from_depot'] = self.distance_matrix[0,i]
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


    def compute_VRP(self, deliveries_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):
        # Generate an initial feasible solution through the Solomon heuristic
        sol = self.constructiveVRP(deliveries_to_do, vehicles_dict)
        
        # ALNS Implementation
        best_sol = self.ALNS_VRP(sol)

        return [s['path'] for s in best_sol]

    def constructiveVRP(self, deliveries_to_do, vehicles_dict):
        for d in self.delivery:
            if d in deliveries_to_do:
                self.delivery[d]['crowdshipped'] = False
            else:
                self.delivery[d]['crowdshipped'] = True

        for d in self.delivery:
            self.delivery[d]['chosen_vrp'] = False

        # sort self.delivery based on their distance from the depot
        self.delivery = dict(sorted(self.delivery.items(), key=lambda x:x[1]['dist_from_depot']))

        sol = []
        for k in range(len(vehicles_dict)):
            # Initialize the solution for the k-th vehicle.
            sol.append({'path': [],'arrival_times':[], 'waiting_times': [],
                'vol_left': vehicles_dict[k]['capacity']}) 
            
            # add to the solution of this vehicle the closest delivery that:
            #   - is still available 
            #   - respects the capacity constraint
            #   - corresponds to an arrival time that is lower
            #     than the upper limit of its time window
            aval_d = [d for _,d in self.delivery.items() if (self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and \
                    self.distance_matrix[0,d['index']] < d['time_window_max']) and \
                    d['crowdshipped'] == False]

            # DEBUG
            #print(f"[DEBUG] distance from depot to 4: {self.distance_matrix[0,4]}")
            #print(f"[DEBUG] aval_d: {aval_d}")
            if aval_d:
                # add the depot as first node in the solution
                sol[k]['path'].append(0)
                sol[k]['arrival_times'].append(0)
                sol[k]['waiting_times'].append(0)

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
                sol[k]['waiting_times'].append(0)

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
                for d in [d for _,d in self.delivery.items() if self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and d['crowdshipped'] == False]:
                    # Find the best position of the delivery d among every
                    # pair of deliveries already in the solution.
                    # Each iteration of this loop considers a different positioning
                    # of the delivery "d", where "prev_n" and "next_n" are the indiced of 
                    # the nodes that precede and follow "d".
                    
                    # Initialize the list containing the best positioning for "d" in terms
                    # of previous / following nodes and cost function c1.
                    #   best_pos_d = [<prev_n>, <next_n>, <c1>]
                    # where <prev_n> and <next_n> are given as indices in the sol[k] lists.
                    best_pos_d = []
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
                            if not best_pos_d: # first time
                                best_pos_d = [prev_n, next_n, c1]
                            elif c1 < best_pos_d[2]: # found a better placing
                                # update the min
                                best_pos_d = [prev_n, next_n, c1]
                    if best_pos_d: # if a best placing was found, add it to "best_pos_all"
                        best_pos_all[d['id']] = best_pos_d

                # Choose which one of the nodes d (for which a best placing inside this path was found)
                # will be included in the path of the k-th vehicle. 
                # The choice is based on the cost function C2.
                if feasible_nodes_flag:
                    # DEBUG
                    #print("[DEBUG} best_pos_all:\n\t")
                    #print(best_pos_all)
                    '''
                    k = [k for k in best_pos_all.keys()]
                    print(f"[DEBUG] id type in best_pos_all: {type(k[0])}")
                    '''

                    best_d_id = self.getBestDelivery(sol[k], best_pos_all)
                    # DEBUG
                    #print(f"[DEBUG] best_d_id: {best_d_id}")

                    # 1) Add the new delivery before the depot at the end of the path
                    # 2) insert the new arrival and waiting times
                    # 3) update all the arrival & waiting times of the following deliveries
                    # 4) update the volume left in the vehicle
                    self.updatePath(sol[k], best_d_id, best_pos_all)
                    
                    # 5) set the chosen_vrp of the delivery to True
                    self.delivery[best_d_id]['chosen_vrp'] = True

            # DEBUG
            if len(sol[k]['path']) > 0:
                print(f"Vehicle n. {k}")
                for n_ind in range(len(sol[k]['path'])):
                    n_id = sol[k]['path'][n_ind]
                    if n_id != 0:
                        print("Node ID\t|\tArrival Time\t|\tWaiting Time\t|\tLower bound\t|\tUpper bound")
                        print(f"{n_id}\t|\t{sol[k]['arrival_times'][n_ind]}\t|\t{sol[k]['waiting_times'][n_ind]}\t|\t{self.delivery[n_id]['time_window_min']}\t|\t{self.delivery[n_id]['time_window_max']}")
                print()

        return sol

    def ALNS_VRP(self, sol):
        curr_sol = best_sol = sol
        curr_sol_paths = [s['path'] for s in curr_sol]
        best_obj = curr_obj = self.env.evaluate_VRP(curr_sol_paths)
        
        # Temperature of the solution
        T = T_start = -self.alns_mu / np.log(0.5) 

        # i: iteration counter
        # j: counter of the iterations without an improvement
        i = j = 0
        while i < self.alns_N_max and j < self.alns_N_IwI:
            # select a destroy operator d according to their weights and apply it to the solution
            d = np.random.choice(list(self.destroy_algos.keys()), p=[self.destroy_algos[dd]['p'] for dd in self.destroy_algos])
            self.destroy_algos[d]['n'] += 1
            sol_minus = self.destroy_algos[d]['func'](sol)

            # select a repair operator r according to their weights and apply it to the solution
            r = np.random.choice(list(self.repair_algos.keys()), p=[self.repair_algos[rr]['p'] for rr in self.repair_algos])
            self.repair_algos[r]['n'] += 1
            sol_plus = self.repair_algos[r]['func'](sol_minus)

            new_sol_paths = [s['path'] for s in sol_plus]
            new_obj = self.env.evaluate_VRP(new_sol_paths)

            if new_obj < curr_obj: 
                # Improvement in the CURRENT solution
                curr_sol = sol_plus
                curr_obj = new_obj
                # increment the score of the used operators by sigma2
                self.destroy_algos[d]['s'] += self.alns_sigma2
                self.repair_algos[r]['s'] += self.alns_sigma2
            else:
                # the current solution MAY be updated even though it has not improved
                v = np.exp(-(new_obj - curr_obj)/T)
                rn = np.random.uniform()
                if rn < v:
                    curr_sol = sol_plus
                    curr_obj = new_obj
                    # increment the score of the used operators by sigma3
                    self.destroy_algos[d]['s'] += self.alns_sigma3
                    self.repair_algos[r]['s'] += self.alns_sigma3
            if new_obj < best_obj:
                # Improvement with respect to the best solution
                best_sol = sol_plus
                best_obj = new_obj
                # increment the score of the used operators by sigma1
                self.destroy_algos[d]['s'] += self.alns_sigma1 # TODO or += self.alns_sigma1 - self.alns_sigma2 ??
                self.repair_algos[r]['s'] += self.alns_sigma1
                j = 0 # reset the counter of iterations without improvement
            else:
                j += 1
            
            # Update the probabilities of the operators using the adaptive weight proceudure
            if i % self.alns_N_s == 0:
                # Evaluate the new weights for the used operators.
                # Then, reset their scores and number of used times.
                for rr in self.repair_algos:
                    if self.repair_algos[rr]['n'] > 0:
                        self.repair_algos[rr]['w'] = (1-self.alns_rho)*self.repair_algos[rr]['w'] + \
                            self.alns_rho*self.repair_algos[rr]['s']/self.repair_algos[rr]['n']

                    self.repair_algos[rr]['n'] = 0
                    self.repair_algos[rr]['s'] = 0

                for dd in self.destroy_algos:
                    if self.destroy_algos[dd]['n'] > 0:
                        self.destroy_algos[dd]['w'] = (1-self.alns_rho)*self.destroy_algos[dd]['w'] + \
                            self.alns_rho*self.destroy_algos[dd]['s']/self.destroy_algos[dd]['n']

                    self.destroy_algos[dd]['n'] = 0
                    self.destroy_algos[dd]['s'] = 0
                
                # Evaluate the new probabilities for the operators
                sum_w_r = sum([self.repair_algos[tmp_r]['w'] for tmp_r in self.repair_algos])
                for rr in self.repair_algos:
                    self.repair_algos[rr]['p'] = self.repair_algos[rr]['w']/sum_w_r
                sum_w_d = sum([self.destroy_algos[tmp_d]['w'] for tmp_d in self.destroy_algos])
                for dd in self.destroy_algos:
                    self.destroy_algos[dd]['p'] = self.destroy_algos[dd]['w']/sum_w_d

            # reduce the temperature by applying the "cooling rate"
            T = T*self.alns_eps
            i += 1
        
        return best_sol
    
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
        if prev_n_id == 0: # depot (first element in the path)
            dist_prev_d = d['dist_from_depot']
        else:
            prev_n_index = self.delivery[prev_n_id]['index'] # index of prev_n in the distance matrix
            dist_prev_d = self.distance_matrix[prev_n_index, d['index']] # distance between prev_n and d
        
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
            sol_k['waiting_times'][prev_n_sol] + \
            dist_prev_d
        
        # DEBUG
        #print(f"[DEBUG] d: {d['id']} | arr_time_d: {arr_time_d}")
        #print(f"[DEBUG] time_window_max of d: {d['time_window_max']}")
        
        if arr_time_d > d['time_window_max']:
            return False
        
        # Evaluate the first Push-forward PF
        next_n_id = sol_k['path'][next_n_sol]
            
        if next_n_id != 0: # not the depot (last element in the path)
            next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
            dist_d_next = self.distance_matrix[d['index'], next_n_index] # distance between d and next_n
            waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
            PF = (arr_time_d + waiting_time_d + dist_d_next) - sol_k['arrival_times'][next_n_sol]
            
            for next_n_sol in range(next_n_sol, len(sol_k['path'])-1):
                # If PF == 0: time feasibility is guaranteed from this point on. Return true
                # If PF + arrival time exceeds the time window upper bound, return False
                next_n_id = sol_k['path'][next_n_sol]
                next_n_timeupperbound = self.delivery[next_n_id]['time_window_max']
                if PF == 0:
                    return True
                
                # DEBUG
                #if next_n_id == 41 and  sol_k['arrival_times'][next_n_sol] + PF > next_n_timeupperbound:
                #    print("CACCA")

                if sol_k['arrival_times'][next_n_sol] + PF > next_n_timeupperbound:
                    return False
                
                # If it hasn't returned, update PF to check the next delivery in the path.
                # NOTE: if the node after next_n_sol in the path is the depot, stop 
                if next_n_sol + 1 != len(sol_k['path'])-1:
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
        if prev_n_id == 0: # depot (first element in the path)
            dist_prev_d = d['dist_from_depot']
        else:
            prev_n_index = self.delivery[prev_n_id]['index'] # index of prev_n in the distance matrix
            dist_prev_d = self.distance_matrix[prev_n_index, d['index']] # distance between prev_n and d

        next_n_id = sol_k['path'][next_n_sol]
        if next_n_id == 0: # depot (last element in the path)
            next_n_index = 0
            dist_d_next = d['dist_from_depot']
        else:
            next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
            dist_d_next = self.distance_matrix[d['index'], next_n_index] # distance between d and next_n

        if prev_n_id == 0:
            dist_prev_next = self.delivery[next_n_id]['dist_from_depot']
        else:
            dist_prev_next = self.distance_matrix[prev_n_index, next_n_index]

        c11 = dist_prev_d + dist_d_next - self.mu_vrp*dist_prev_next
        
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
                sol_k['waiting_times'][prev_n_sol] + \
                dist_prev_d
        waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        c12 = new_arr_time_next - sol_k['arrival_times'][next_n_sol]

        c1 = self.alpha1_c1_vrp*c11 + self.alpha2_c1_v2p*c12
        return c1

    def getC2(self, d, c1):
        """
        """
        return self.lambda_vrp*d['dist_from_depot'] - c1 + self.volw*d['vol']

    def compareC2(self, c2_first, c2_second):
        """
        Compare two values of the cost function C2 and return True if the 
        first one is better than the second, false otherwise.
        In this implementation, "better" means higher.
        """
        return c2_first > c2_second

    def getBestDelivery(self, sol_k, best_pos_all):
        """
        """
        best_d = [None, 0] # [<id>, <c2>]
        for d_id in best_pos_all:
            c2_d = self.getC2(self.delivery[d_id], best_pos_all[d_id][2])

            #print(f"[DEBUG] c2_d: {c2_d}")

            # compare the cost c2 of the currently selected delivery "d_id"
            # with the optimum one. Update the optimum if better.
            if best_d[0] == None: # first check
                best_d = [d_id, c2_d]
            elif self.compareC2(c2_d, best_d[1]):
                best_d = [d_id, c2_d]

        return best_d[0] # return the id of the delivery with optimum c2


    def updatePath(self, sol_k, best_d_id, best_pos_all):
        """
        """
        #print(f"[DEBUG] best_d_id: {best_d_id}") #DEBUG
        prev_n_sol = best_pos_all[best_d_id][0]
        next_n_sol = best_pos_all[best_d_id][1]
        # 1) Add the new delivery in the chosen place
        sol_k['path'].insert(prev_n_sol+1, best_d_id)
        # update the index of next_n
        next_n_sol += 1

        # 2) insert the new arrival and waiting times
        prev_n_id = sol_k['path'][prev_n_sol]
        if prev_n_id == 0: # depot (first element in the path)
            dist_prev_d = self.delivery[best_d_id]['dist_from_depot']
        else:
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
        if next_n_id == 0: # depot (last element in the path)
            dist_d_next = self.delivery[best_d_id]['dist_from_depot']
        else:
            next_n_index = self.delivery[next_n_id]['index'] # index of next_n in the distance matrix
            dist_d_next = self.distance_matrix[self.delivery[best_d_id]['index'], next_n_index] # distance between d and next_n
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        additional_delay_flag = True
        while additional_delay_flag and next_n_sol < len(sol_k['path']):
            old_arr_time_next = sol_k['arrival_times'][next_n_sol]
            # Update the arrival time at next_n
            sol_k['arrival_times'][next_n_sol] = new_arr_time_next
            if next_n_sol != len(sol_k['path'])-1: # NOT the depot
                delta_arr_time_next = new_arr_time_next - max(old_arr_time_next, self.delivery[next_n_id]['time_window_min'])
                arr_time_relative = self.delivery[next_n_id]['time_window_min']-new_arr_time_next
                sol_k['waiting_times'][next_n_sol] = max(0, arr_time_relative)
                if arr_time_relative < 0: # if the arrival at next_n is after the lower bound of its time window
                    # Update the arrival time at the delivery after "next_n" taking into consideration
                    # the delay that was introduced at "next_n"
                    next_n_sol += 1
                    if next_n_sol < len(sol_k['path']):
                        next_n_id = sol_k['path'][next_n_sol]
                        new_arr_time_next = sol_k['arrival_times'][next_n_sol] + delta_arr_time_next
                else: # otherwise, no additional delay has been introduced from next_n on in the path: exit the while loop 
                    additional_delay_flag = False
            else:
                next_n_sol +=1
        
        # 4) update the volume left in the vehicle
        sol_k['vol_left'] -= self.delivery[best_d_id]['vol']


    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
