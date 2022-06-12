# -*- coding: utf-8 -*-
import time
import copy
from agents import *
from scipy import spatial
import numpy as np

class HeuGroup18(Agent):

    def __init__(self, env):
        self.name = "HeuGroup2"
        self.env = env
        # evaluate the distance matrix
        self.distance_matrix = self.env.distance_matrix
        self.delivery = []
        self.init_sol_created = False
        self.learning_flag = False
        self.n_crowdshipped = 0 # number of crowdshipped deliveries

        # note: beta1 + beta2 must be 1 and each of them must be >= 0
        self.beta1_c3 = 0.5
        self.beta2_c3 = 1-self.beta1_c3
        self.mu_vrp = 1
        self.lambda_vrp = 1
        self.volw = 1 # weight associated to the volume of the delivery

        # ALNS Parameters
        self.alns_N_max = 10000 # max number of iterations
        self.alns_N_IwI = 5000 # max number of iterations without an improvement
        self.alns_N_s = 50 # number of iterations in a segment
        self.alns_mu = 0.05 # tuning parameter for the "temperature" of a solution
        self.alns_eps = 0.9998  # cooling rate for the temperature
        # parameters used to increment the scores of the operators
        self.alns_sigma1 = 20 # if the new sol is better than the best one
        self.alns_sigma2 = 16 # if the new sol is better than the curr one
        self.alns_sigma3 = 13 # if the new sol is NOT better than the curr one but it is chosen
        self.alns_rho = 0.1 # "reaction factor" used to update the weights of the operators
        self.alns_p = 1 # degree of "randomness" used in the alns algorithms. p >= 1. p = 1: random choice

        self.vehicles_dict = []
        self.vehicles_order = []
        self.veh_p = 1 # degree of "randomness" used in the generation of vehicles permutations

        # Dictionaries used in the learning phase
        self.data_improv_volw = {
            "volw": [],
            "obj": []
        }
        self.data_improv_alns_eps = {
            "alns_eps": [],
            "obj": []
        }
        self.data_improv_beta1_c3 = {
            "beta1_c3": [],
            "obj": []
        }
        self.data_improv_alns_p = {
            "alns_p": [],
            "obj": []
        }
    
    # ALNS Heuristics
    def alns_repair_greedy(self, sol): 
        # 1) For each vehicle, find the best node (in terms of c2) to insert in the best position (in terms of c1)
        # Format used:
        #   best_pos_ve = {<vehicle_index>: [<best_pos_k of that vehicle>]}
        best_pos_ve = {}
        # Define the maximum distance from depot considering the nodes that are still not in a solution
        # It will be needed to normalize the c1 coefficient when we are inserting a node in an empty vehicle
        avail_nodes = [d['dist_from_depot'] for _,d in self.delivery.items() if d['chosen_vrp'] == False and d['crowdshipped'] == False]
        if avail_nodes:
            max_dist_depot = max(avail_nodes)

        for k in range(len(sol)):
            # best node in terms of c2 for vehicle k. 
            # Format: best_pos_k = [<prev_n>,<next_n>,<c1>,<c2>,<node_id>]
            best_pos_k = [] 
            for d in [d for _,d in self.delivery.items() if self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and d['crowdshipped'] == False]:
                # find the best position where to insert node d in vehicle k
                # best_pos_d = [<prev_node_index>,<next_node_index>,<c1>]
                best_pos_d = []
                for i in range(len(sol[k]['path'])-1):
                    prev_n = i
                    next_n = i+1
                    # Check time feasibility considering this insertion
                    if self.nodeTimeFeasibleVRP(sol[k], prev_n, d, next_n):
                        c1 = self.getC1(sol[k], prev_n, d, next_n)
                        if not best_pos_d: # first time
                            best_pos_d = [prev_n, next_n, c1]
                        elif c1 < best_pos_d[2]:
                            best_pos_d = [prev_n, next_n, c1]
                if best_pos_d: # if a best placing was found, add it to "best_pos_all"
                    if not best_pos_k:
                        c2_new = self.getC2(d, best_pos_d[2])
                        best_pos_d.append(c2_new)
                        best_pos_d.append(d['id'])
                        best_pos_k = best_pos_d
                    else:
                        c2_new = self.getC2(d, best_pos_d[2])
                        if self.compareC2(c2_new, best_pos_k[3]):
                            best_pos_d.append(c2_new)
                            best_pos_d.append(d['id'])
                            best_pos_k = best_pos_d
                else:
                    if sol[k]['n_nodes'] == 0:
                        # empty vehicle: nodes are evaluated on the base of their distance from the depot
                        # (it corresponds to their value of c2). In this case, however, nodes closer to 
                        # the depot will be preferred.
                        # Check time feasibility first.
                        if d['dist_from_depot'] < d['time_window_max']:
                            #TODO FIX DISTANCE ASIMMETRY
                            c1_new = d['dist_from_depot'] / max_dist_depot
                            c2_new = d['dist_from_depot']
                            if not best_pos_k:
                                best_pos_k = [0,1,c1_new,c2_new,d['id']]
                            else:
                                if c2_new < best_pos_k[3]:
                                    best_pos_k = [0,1,c1_new,c2_new,d['id']]

            if best_pos_k:
                best_pos_ve[k] = best_pos_k

        # 2) While there are still vehicles with a best insertion available:
        #       - Find the vehicle whose insertion has the best cost (in terms of c3)
        #       - Perform the insertion
        #       - Update the best insertion of all vehicles whose best insertion was the inserted node
        
        while best_pos_ve: 
            # List that will contain the best vehicle in terms of the c3 cost function, defined as:
            #   c3(v) = c1(best_node(v)) + (volume of best_node(v))/(sum of volumes of all nodes in v)
            # The format used is: 
            #   best_ve_c3 = [<index of the vehicle in best_pos_ve>, <value of c3 for that vehicle>]
            # Note that "best" for the c3 cost function means MINIMUM.
            best_ve_c3 = []
            for v in best_pos_ve:
                c3 = self.getC3(sol[v], best_pos_ve[v])
                if not best_ve_c3:
                    best_ve_c3 = [v, c3]
                else:
                    if c3 < best_ve_c3[1]:
                        best_ve_c3 = [v, c3]
            
            # Perform the insertion
            new_id = best_pos_ve[best_ve_c3[0]][4]
            new_prev_n = best_pos_ve[best_ve_c3[0]][0]
            new_next_n = best_pos_ve[best_ve_c3[0]][1]

            self.insertNode(sol[best_ve_c3[0]], new_id, new_prev_n, new_next_n)

            # Repeat the same procedure done before, but this time only for those vehicles 
            # whose best node was the same that we inserted in the last line of code
            avail_nodes = [d['dist_from_depot'] for _,d in self.delivery.items() if d['chosen_vrp'] == False and d['crowdshipped'] == False]
            if avail_nodes:
                max_dist_depot = max(avail_nodes)
            for k in range(len(sol)):
                if k in best_pos_ve:
                    if best_pos_ve[k][4] == new_id:
                        best_pos_ve.pop(k)
                        # best node in terms of c2 for vehicle k. 
                        # Format: best_pos_k = [<prev_n>,<next_n>,<c1>,<c2>,<node_id>]
                        best_pos_k = [] 
                        for d in [d for _,d in self.delivery.items() if self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and d['crowdshipped'] == False]:
                            # find the best position where to insert node d in vehicle k
                            # best_pos_d = [<prev_node_index>,<next_node_index>,<c1>]
                            best_pos_d = []
                            for i in range(len(sol[k]['path'])-1):
                                prev_n = i
                                next_n = i+1
                                # Check time feasibility considering this insertion
                                if self.nodeTimeFeasibleVRP(sol[k], prev_n, d, next_n):
                                    c1 = self.getC1(sol[k], prev_n, d, next_n)
                                    if not best_pos_d: # first time
                                        best_pos_d = [prev_n, next_n, c1]
                                    elif c1 < best_pos_d[2]:
                                        best_pos_d = [prev_n, next_n, c1]
                            if best_pos_d: 
                                if not best_pos_k:
                                    c2_new = self.getC2(d, best_pos_d[2])
                                    best_pos_d.append(c2_new)
                                    best_pos_d.append(d['id'])
                                    best_pos_k = best_pos_d
                                else:
                                    c2_new = self.getC2(d, best_pos_d[2])
                                    if self.compareC2(c2_new, best_pos_k[3]):
                                        best_pos_d.append(c2_new)
                                        best_pos_d.append(d['id'])
                                        best_pos_k = best_pos_d
                            else:
                                if sol[k]['n_nodes'] == 0:
                                    # empty vehicle: nodes are evaluated on the base of their distance from the depot
                                    # (it corresponds to their value of c2). In this case, however, nodes closer to 
                                    # the depot will be preferred.
                                    # Check time feasibility first.
                                    if d['dist_from_depot'] < d['time_window_max']:
                                        #TODO FIX DISTANCE ASIMMETRY
                                        c1_new = d['dist_from_depot'] / max_dist_depot
                                        c2_new = d['dist_from_depot']
                                        if not best_pos_k:
                                            best_pos_k = [0,1,c1_new,c2_new,d['id']]
                                        else:
                                            if c2_new < best_pos_k[3]:
                                                best_pos_k = [0,1,c1_new,c2_new,d['id']]

                        if best_pos_k:
                            best_pos_ve[k] = best_pos_k
        return sol
        
    def alns_repair_regret(self, sol):
        """
        Insert as many nodes as possible in the solution. 
        In each iteration, the node whose "regret value"
        is the highest will be inserted in the minimum cost position.
        The insertions go on until no more feasible insertions are found.
        In the regret-2 implementation, the regret value is defined as
        the cost difference between inserting the node in the second 
        best solution compared to the best solution.
        """
        ins_avail_flag = True # there are still nodes available for insertion
        while ins_avail_flag:
            # Initialize the list that will contain the insertion with
            # the best (highest) regret value among all the insertions that
            # will be found. 
            best_ins_regretval = [] 
            # Initialize also the best regret value
            best_regretval = None
            # Nodes that can be inserted in one vehicle only will have the priority.
            # This flag will be set to True when one of such nodes is found
            one_vehicle_flag = False

            # Define the maximum distance from depot considering the nodes that are still not in a solution
            # It will be needed to normalize the c1 coefficient when we are inserting a node in an empty vehicle
            avail_nodes = [d['dist_from_depot'] for _,d in self.delivery.items() if d['chosen_vrp'] == False and d['crowdshipped'] == False]
            if avail_nodes:
                max_dist_depot = max(avail_nodes)

            # list of nodes not yet in a solution and not crowdshipped
            for d in [d for _,d in self.delivery.items() if d['chosen_vrp'] == False and d['crowdshipped'] == False]:
                '''
                best_ins_d: best vehicle where to insert d in terms of c3 
                    Structure: [
                        <vehicle_number>,
                        <prev_n>,
                        <next_n>,
                        <c1 of the insertion>,
                        <node_id>,
                        <c3 of the insertion>
                    ]
                
                second_best_d: second best vehicle where to insert d in terms of c3
                '''
                best_ins_d = [] 
                second_best_ins_d = [] 
                # consider each vehicle
                for k in range(len(sol)):
                    # check if the node d would fit in the vehicle k
                    if d['vol'] <= sol[k]['vol_left']:
                        # control each possible position where to insert d in k
                        best_pos_d = []
                        for i in range(len(sol[k]['path'])-1):
                            prev_n = i
                            next_n = i+1
                            #check time feasibilty 
                            if self.nodeTimeFeasibleVRP(sol[k],prev_n,d,next_n):
                                c1 = self.getC1(sol[k], prev_n, d, next_n)
                                
                                if not best_pos_d:
                                    best_pos_d = [prev_n, next_n, c1]
                                elif c1 < best_pos_d[2]:
                                    best_pos_d = [prev_n, next_n, c1]
                        if best_pos_d:
                            c3 = self.getC3(sol[k], [
                                best_pos_d[0],
                                best_pos_d[1],
                                best_pos_d[2],
                                0, # c2 is not considered
                                d['id']
                            ])
                            if not best_ins_d:  
                                best_ins_d = [
                                    k,
                                    best_pos_d[0],
                                    best_pos_d[1],
                                    best_pos_d[2],
                                    d['id'],
                                    c3
                                ]
                            elif not second_best_ins_d:
                                second_best_ins_d = [
                                    k,
                                    best_pos_d[0],
                                    best_pos_d[1],
                                    best_pos_d[2],
                                    d['id'],
                                    c3
                                ]
                            else:
                                if c3 < second_best_ins_d[5]:
                                    if c3 < best_ins_d[5]:
                                        # new best insertion found for this node
                                        best_ins_d = [
                                            k,
                                            best_pos_d[0],
                                            best_pos_d[1],
                                            best_pos_d[2],
                                            d['id'],
                                            c3
                                        ]
                                    else:
                                        # new second best insertion found for this node
                                        second_best_ins_d = [
                                            k,
                                            best_pos_d[0],
                                            best_pos_d[1],
                                            best_pos_d[2],
                                            d['id'],
                                            c3
                                        ]
                        else: # referred to if best_pos_d
                            # check for empty vehicles
                            if sol[k]['n_nodes'] == 0:
                                # Check time feasibility first.
                                if d['dist_from_depot'] < d['time_window_max']:
                                    c1_new = d['dist_from_depot'] / max_dist_depot
                                    c3 = self.getC3(sol[k],[0,1,c1_new,0,d['id']])
                                    if not best_ins_d:
                                        best_ins_d = [
                                            k,
                                            0,
                                            1,
                                            c1_new,
                                            d['id'],
                                            c3
                                        ]
                                    elif not second_best_ins_d:
                                        second_best_ins_d = [
                                            k,
                                            0,
                                            1,
                                            c1_new,
                                            d['id'],
                                            c3
                                        ]
                                    else:
                                        if c3 < second_best_ins_d[5]:
                                            if c3 < best_ins_d[5]:
                                                # new best insertion found for this node
                                                best_ins_d = [
                                                    k,
                                                    0,
                                                    1,
                                                    c1_new,
                                                    d['id'],
                                                    c3
                                                ]
                                            else:
                                                # new second best insertion found for this node
                                                second_best_ins_d = [
                                                    k,
                                                    0,
                                                    1,
                                                    c1_new,
                                                    d['id'],
                                                    c3
                                                ]
                                
                if not second_best_ins_d:
                    if not one_vehicle_flag:
                        one_vehicle_flag = True
                        best_ins_regretval = best_ins_d.copy()
                    else: # not the first node which has only one vehicle where it can be inserted
                        # Solve the tie by comparing the c3 values
                        if best_ins_d[5] < best_ins_regretval[5]:
                            best_ins_regretval = best_ins_d.copy()
                else: # nodes which can be inserted in (at least) two vehicles
                    if not one_vehicle_flag:
                        # Evaluate the regret value
                        regretval = second_best_ins_d[5] - best_ins_d[5]
                        if not best_ins_regretval:
                            best_ins_regretval = best_ins_d.copy()
                            best_regretval = regretval
                        elif regretval > best_regretval:
                            best_ins_regretval = best_ins_d.copy()
                            best_regretval = regretval

            # all nodes have been considered...
            if best_ins_regretval:
                # if a best insertion in terms of regret value has been found, 
                # perform it.
                self.insertNode(
                    sol[best_ins_regretval[0]],
                    best_ins_regretval[4],
                    best_ins_regretval[1],
                    best_ins_regretval[2],
                )
            else:
                ins_avail_flag = False

        return sol
                        
    def alns_destroy_worst(self, sol):
        """
        Remove q nodes from the solution. The node removed in each one of the q 
        iterations is the one associated to the highest cost in terms of c3 (a cost function).
        """
        q = max(1,min(int(self.env.n_deliveries / 10), 25))
        for i in range(q):
            # Create a list called "deliv" which contains pairs of the type [<node_id>,<vehicle # of the node>]
            deliv = []
            for v in range(len(sol)):
                for n in sol[v]['path']:
                    if n != 0:
                        deliv.append([n,v])

            # sort "deliv" on the basis of the cost function c3 evaluated for each delivery
            worst_deliv = sorted(deliv, key= lambda x: \
                    self.getC3(
                        sol[x[1]],
                        [
                            sol[x[1]]['path'].index(x[0])-1,
                            sol[x[1]]['path'].index(x[0])+1,
                            self.getC1(
                                sol[x[1]],
                                sol[x[1]]['path'].index(x[0])-1,
                                self.delivery[str(x[0])],
                                sol[x[1]]['path'].index(x[0])+1
                            ),
                            0,
                            x[0]
                        ]
                    ), reverse=True)
            
            # choose a random number y in the interval [0,1]
            y = np.random.uniform()
            # Choose which delivery will be removed. Notice that the removal is randomized, 
            # with the degree of randomization controlled by the parameter p
            worst_d = worst_deliv[int(np.power(y,self.alns_p)*(len(worst_deliv)-1))]
            self.removeNode(sol[worst_d[1]], worst_d[0], 
                sol[worst_d[1]]['path'].index(worst_d[0])-1,
                sol[worst_d[1]]['path'].index(worst_d[0])+1
            )

        return sol
               
    def alns_destroy_random(self, sol):
        """
        Remove q random nodes from the solution
        """
        q = max(1,min(int(self.env.n_deliveries / 10), 25))
        for i in range(q):
            # pick a random vehicle
            v = np.random.randint(0,len(sol))
            # pick a random node in the path of the picked vehicle, excluding the depot (first and last elements)
            if sol[v]['n_nodes'] > 0:
                n = np.random.randint(1,len(sol[v]['path'])-1) 
                n_id = sol[v]['path'][n]
                self.removeNode(sol[v], n_id, n-1, n+1)
            else:
                i -= 1 # repeat the iteration if an empty vehicle was picked
        return sol

    def compute_delivery_to_crowdship(self, deliveries):
        # 1) evaluate the score for all deliveries
        if len(deliveries) == 0:
            return []
        self.delivery = deliveries

        vehicles_dict = self.env.get_vehicles()
        #alns_N_max = 8000
        #alns_N_IwI = 800
        n_it = 10 # num of iterations
        # Generate a first VRP solution (simplified VRP, less iterations) with no
        # nodes in crowdshipping
        #self.init_sol_created = False
        self.n_crowdshipped = 0
        VRP_solution_init = self.compute_VRP(self.env.get_delivery(), self.env.get_vehicles())
        obj_init = self.env.evaluate_VRP(VRP_solution_init)
        #DEBUG
        print(f"[DEBUG] OBJ FUNC BEFORE CROWDSHIPPING: {obj_init}")
        # Create a list called "deliv" which contains triplets of the type:
        #    [<node_id>,<vehicle # of the node>,<number of times the node is proposed for crowdshipping>]
        deliv = []
        for v in range(len(VRP_solution_init)):
            for n in VRP_solution_init[v]:
                if n != 0:
                    deliv.append([n,v,0])
        
        for i in range(n_it):
            # save the VRP solution
            VRP_sol_curr = copy.deepcopy(VRP_solution_init)
            # consider each node, evaluate the obj func if it is removed and
            # compare the cost gain with its crowshipping (stochastic) cost
            np.random.shuffle(deliv)
            obj_curr = obj_init
            for d in deliv:
                n_index = VRP_sol_curr[d[1]].index(d[0]) # index of the node in its vehicle
                VRP_sol_curr[d[1]].remove(d[0])
                if VRP_sol_curr[d[1]] == [0,0]: # vehicle became empty
                    VRP_sol_curr[d[1]] = []
                # evaluate the cost "gain" obtained by removing the node
                obj_new = self.env.evaluate_VRP(VRP_sol_curr)
                df = obj_curr - obj_new
                sum_vol_nodes = self.delivery[str(d[0])]['vol'] + sum([self.delivery[dd]['vol'] for dd in self.delivery if self.delivery[dd]['id'] in VRP_sol_curr[d[1]]])
                if VRP_sol_curr[d[1]] == [0,0]: # vehicle became empty
                    c3 = df
                else:
                    c3 = df + (self.delivery[str(d[0])]['vol']/sum_vol_nodes)*vehicles_dict[d[1]]['cost']
                # compare the insertion cost c3 with the stochastic crowdshipping cost
                if c3 > self.delivery[str(d[0])]['crowd_cost']:
                    # Better to try to crowdship this node. 
                    d[2] += 1 # increment the node's counter
                    # Note that the node is considered
                    # as crowdshipped by comparing a randomly chosen probability with the p_failed
                    # of such node.
                    y = np.random.uniform()
                    if y >= self.delivery[str(d[0])]['p_failed']: 
                        # crowshipping failed: put the node back into the solution
                        if not VRP_sol_curr[d[1]]:
                            VRP_sol_curr[d[1]] = [0,0]
                        VRP_sol_curr[d[1]].insert(n_index, d[0])
                    else:
                        obj_curr = obj_new
                else:
                    # the node shouldn't be proposed for crowdshipping: put it back into the solution
                    if not VRP_sol_curr[d[1]]:
                            VRP_sol_curr[d[1]] = [0,0]
                    VRP_sol_curr[d[1]].insert(n_index, d[0])


        # 3) Propose for crowdshipping those deliveries that were proposed for crowdshipping in the 
        # previous for loop in more than half of the iterations
        id_to_crowdship = [str(d[0]) for d in deliv if d[2]/n_it >= 0.5]

        return id_to_crowdship 

    def compute_VRP(self, deliveries_to_do, vehicles_dict, alns_N_max=None, alns_N_IwI=None):
        if self.vehicles_dict:
            vehicles_dict = self.vehicles_dict
        
        # DEBUG
        #self.vehicles_order = list(range(0, len(vehicles_dict)))

        # Generate an initial feasible solution through the Solomon heuristic
        if not self.learning_flag:
            if not self.init_sol_created:
                self.sol = self.constructiveVRP(deliveries_to_do, vehicles_dict)
                sol = self.sol
                self.init_sol_created= True
            else:
                sol = self.sol
                sol_copy = copy.deepcopy(sol)
                # remove the nodes that were crowdshipped
                for v in range(len(sol_copy)):
                    q = 1
                    q_copy = 1
                    if sol_copy[v]['path']:
                        while sol_copy[v]['path'][q_copy] != 0:
                            if str(sol_copy[v]['path'][q_copy]) not in deliveries_to_do:
                                # this node has been crowdshipped: remove it from sol
                                self.removeNode(sol[v], sol[v]['path'][q], q-1, q+1)
                                self.delivery[str(sol_copy[v]['path'][q_copy])]['crowdshipped'] = True
                                self.delivery[str(sol_copy[v]['path'][q_copy])]['chosen_vrp'] = False
                                self.n_crowdshipped += 1
                                q -= 1
                            q += 1
                            q_copy += 1
        else:
            sol = self.constructiveVRP(deliveries_to_do, vehicles_dict)
        
        # ALNS Implementation
        # Returns the best solution overall in terms of objetive function and the best function found which
        # also includes all deliveries.
        if not alns_N_max:
            alns_N_max = self.alns_N_max
        if not alns_N_IwI:
            alns_N_IwI = self.alns_N_IwI
        best_sol, best_sol_allnodes = self.ALNS_VRP(sol, alns_N_max, alns_N_IwI)

        # TODO check this part 
        # If best_sol does NOT contain all the nodes not crowdshipped but a solution containing
        # ALL nodes has been found in the ALNS, then use the latter as best solution
        '''
        n_nodes_sol = sum([s['n_nodes'] for s in best_sol])
        if n_nodes_sol < self.env.n_deliveries - self.n_crowdshipped and best_sol_allnodes:
            best_sol = best_sol_allnodes
        '''

        """ for k in range(len(best_sol)):
            # DEBUG
            if len(sol[k]['path']) > 0:
                print(f"Vehicle n. {k}")
                for n_ind in range(len(sol[k]['path'])):
                    n_id = sol[k]['path'][n_ind]
                    if n_id != 0:
                        print("Node ID\t|\tArrival Time\t|\tWaiting Time\t|\tLower bound\t|\tUpper bound")
                        print(f"{n_id}\t|\t{sol[k]['arrival_times'][n_ind]}\t|\t{sol[k]['waiting_times'][n_ind]}\t|\t{self.delivery[str(n_id)]['time_window_min']}\t|\t{self.delivery[str(n_id)]['time_window_max']}")
                print() """ 

        # Revert the order of vehicles to the one used in the original scheme
        # then, return the best solution
        return [s['path'] for s in self.restore_vehicles_order(best_sol)]

    def constructiveVRP(self, deliveries_to_do, vehicles_dict):
        if not self.delivery:
            self.delivery = deliveries_to_do

        for d in self.delivery:
            # evaluate the distance of every delivery from the depot
            self.delivery[d]['dist_from_depot'] = self.distance_matrix[0,self.delivery[d]['id']]
            self.delivery[d]['chosen_vrp'] = False
            if d in deliveries_to_do:
                self.delivery[d]['crowdshipped'] = False
            else:
                self.delivery[d]['crowdshipped'] = True

        # sort self.delivery based on their distance from the depot
        self.delivery = dict(sorted(self.delivery.items(), key=lambda x:x[1]['dist_from_depot']))

        sol = []
        for k in range(len(vehicles_dict)):
            # Initialize the solution for the k-th vehicle.
            sol.append({'path': [],'arrival_times':[], 'waiting_times': [],
                'vol_left': vehicles_dict[k]['capacity'], 
                'init_vol': vehicles_dict[k]['capacity'], 'n_nodes': 0,
                'cost': vehicles_dict[k]['cost']}) 
            
            # add to the solution of this vehicle the closest delivery that:
            #   - is still available 
            #   - respects the capacity constraint
            #   - corresponds to an arrival time that is lower
            #     than the upper limit of its time window
            aval_d = [d for _,d in self.delivery.items() if (self.nodeIsFeasibleVRP(d, sol[k]['vol_left']) and \
                    self.distance_matrix[0,d['id']] < d['time_window_max']) and \
                    d['crowdshipped'] == False]

            # DEBUG
            #print(f"[DEBUG] distance from depot to 4: {self.distance_matrix[0,4]}")
            #print(f"[DEBUG] aval_d: {aval_d}")
            if aval_d:
                #TODO We could use the new insertNode() function
                # directly here to add the first node. It would
                # handle everything itself.

                # add the depot as first node in the solution
                sol[k]['path'].append(0)
                sol[k]['arrival_times'].append(0)
                sol[k]['waiting_times'].append(0)

                sol[k]['path'].append(aval_d[0]['id'])
                sol[k]['arrival_times'].append(self.distance_matrix[0,aval_d[0]['id']])
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
                sol[k]['n_nodes'] += 1

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
                    best_d_id = self.getBestDelivery(sol[k], best_pos_all)

                    # 1) Add the new delivery before the depot at the end of the path
                    # 2) insert the new arrival and waiting times
                    # 3) update all the arrival & waiting times of the following deliveries
                    # 4) update the volume left in the vehicle
                    # 5) increment the number of nodes in the vehicle
                    self.insertNode(sol[k], best_d_id, best_pos_all[best_d_id][0], best_pos_all[best_d_id][1])

            # DEBUG
            '''
            if len(sol[k]['path']) > 0:
                print(f"Vehicle n. {k}")
                for n_ind in range(len(sol[k]['path'])):
                    n_id = sol[k]['path'][n_ind]
                    if n_id != 0:
                        print("Node ID\t|\tArrival Time\t|\tWaiting Time\t|\tLower bound\t|\tUpper bound")
                        print(f"{n_id}\t|\t{sol[k]['arrival_times'][n_ind]}\t|\t{sol[k]['waiting_times'][n_ind]}\t|\t{self.delivery[n_id]['time_window_min']}\t|\t{self.delivery[n_id]['time_window_max']}")
                print()
            '''

        return sol

    def ALNS_VRP(self, sol, alns_N_max, alns_N_IwI):
        # Repair algorithms (ALNS)
        #   'p': probability of the algorithm
        #   'w': weight of the algorithm
        #   's': score of the algorithm in the current segment (used to evaluate the weight)
        #   'n': number of times the algorithm was chosen in the current segment
        self.repair_algos = {
            'greedy': {'func': self.alns_repair_greedy, 'p': 0.5, 'w': 1, 's': 0, 'n': 0},
            'regret': {'func': self.alns_repair_regret, 'p': 0.5, 'w': 1, 's': 0, 'n': 0},
            #'random': {'func': self.alns_repair_rand_choice, 'p': 0.33, 'w': 1, 's': 0, 'n': 0}
        }

        # Destroy algorithms (ALNS)
        self.destroy_algos = {
            #'shaw': {'func': self.alns_destroy_shaw, 'p': 0.33, 'w': 1, 's': 0, 'n': 0},
            'worst': {'func': self.alns_destroy_worst, 'p': 0.5, 'w': 1, 's': 0, 'n': 0},
            'random': {'func': self.alns_destroy_random, 'p': 0.5, 'w': 1, 's': 0, 'n': 0}
        }

        best_sol_allnodes = [] # keep track of the best solution with ALL nodes connected
        curr_sol = copy.deepcopy(sol)
        best_sol = copy.deepcopy(sol)
        curr_sol_paths = [s['path'] for s in self.restore_vehicles_order(curr_sol)]
        best_obj = curr_obj = self.env.evaluate_VRP(curr_sol_paths)/(sum([s['n_nodes'] for s in curr_sol]))
        
        # Temperature of the solution
        T = -self.alns_mu*curr_obj/ np.log(0.5) 

        # i: iteration counter
        # j: counter of the iterations without an improvement
        i = j = 0
        while i < alns_N_max and j < alns_N_IwI:
            deliv_info_copy = copy.deepcopy(self.delivery)

            # select a destroy operator d according to their weights and apply it to the solution
            d = np.random.choice(list(self.destroy_algos.keys()), p=[self.destroy_algos[dd]['p'] for dd in self.destroy_algos])
            self.destroy_algos[d]['n'] += 1
            sol_minus = self.destroy_algos[d]['func'](copy.deepcopy(curr_sol))

            # select a repair operator r according to their weights and apply it to the solution
            r = np.random.choice(list(self.repair_algos.keys()), p=[self.repair_algos[rr]['p'] for rr in self.repair_algos])
            self.repair_algos[r]['n'] += 1
            sol_plus = self.repair_algos[r]['func'](sol_minus)

            new_sol_paths = [s['path'] for s in self.restore_vehicles_order(sol_plus)]
            new_sol_nnodes = sum([s['n_nodes'] for s in sol_plus])
            new_obj = self.env.evaluate_VRP(new_sol_paths)/new_sol_nnodes

            if new_obj < curr_obj: 
                # Improvement in the CURRENT solution
                curr_sol = copy.deepcopy(sol_plus)
                curr_obj = new_obj
                # increment the score of the used operators by sigma2
                self.destroy_algos[d]['s'] += self.alns_sigma2
                self.repair_algos[r]['s'] += self.alns_sigma2
            else:
                # the current solution MAY be updated even though it has not improved
                v = np.exp(-(new_obj - curr_obj)/T)
                rn = np.random.uniform()
                if rn < v:
                    curr_sol = copy.deepcopy(sol_plus)
                    curr_obj = new_obj
                    # increment the score of the used operators by sigma3
                    self.destroy_algos[d]['s'] += self.alns_sigma3
                    self.repair_algos[r]['s'] += self.alns_sigma3
                else:
                    # self.delivery has to be restored to what it was before the changes
                    # introuced by the destroy/repair algorithms
                    self.delivery = deliv_info_copy
            
            # TODO Delete this part (or fix it to take crowdshipped deliveries into account)
            if new_sol_nnodes == self.env.n_deliveries: # the new solution connects all nodes
                if not best_sol_allnodes: # still not have a best solution with all nodes
                    best_sol_allnodes = copy.deepcopy(sol_plus)
                    best_sol_allnodes_obj = new_obj
                elif new_obj < best_sol_allnodes_obj:
                    # The new solution connects all nodes and is better than the best solution with all nodes
                    best_sol_allnodes = copy.deepcopy(sol_plus)
                    best_sol_allnodes_obj = new_obj

            # Improvement with respect to the best solution
            if new_obj < best_obj:
                best_sol = copy.deepcopy(sol_plus)
                best_obj = new_obj
                # increment the score of the used operators by sigma1
                self.destroy_algos[d]['s'] += self.alns_sigma1 - self.alns_sigma2
                self.repair_algos[r]['s'] += self.alns_sigma1 - self.alns_sigma2
                j = 0 # reset the counter of iterations without improvement
            else:
                j += 1
            
            # Update the probabilities of the operators using the adaptive weight proceudure
            if i != 0 and i % self.alns_N_s == 0:
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
            T = max(0.0001,T*self.alns_eps)
            i += 1
        
        return best_sol, best_sol_allnodes
    
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
        dist_prev_d = self.distance_matrix[prev_n_id, d['id']] # distance between prev_n and d
        
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
            dist_d_next = self.distance_matrix[d['id'], next_n_id] # distance between d and next_n
            waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
            PF = (arr_time_d + waiting_time_d + dist_d_next) - sol_k['arrival_times'][next_n_sol]
            
            for next_n_sol in range(next_n_sol, len(sol_k['path'])-1):
                # If PF == 0: time feasibility is guaranteed from this point on. Return true
                # If PF + arrival time exceeds the time window upper bound, return False
                next_n_id = sol_k['path'][next_n_sol]
                next_n_timeupperbound = self.delivery[str(next_n_id)]['time_window_max']
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
                    PF = max(0, PF - sol_k['waiting_times'][next_n_sol])

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

        #c1 is the sum of two components:
        #1) the difference of the new arrival time in next and the old arrival time in next
        #2) the new waiting time in next

        prev_n_id = sol_k['path'][prev_n_sol]
        dist_prev_d = self.distance_matrix[prev_n_id, d['id']] # distance between prev_n and d

        next_n_id = sol_k['path'][next_n_sol]
        dist_d_next = self.distance_matrix[d['id'], next_n_id] # distance between d and next_n
        
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
                sol_k['waiting_times'][prev_n_sol] + \
                dist_prev_d
        waiting_time_d = max(0, d['time_window_min'] - arr_time_d)
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        if next_n_id == 0:
            waiting_time_next = 0
        else:
            waiting_time_next = max(0,self.delivery[str(next_n_id)]['time_window_min'] - new_arr_time_next) #new waiting time of next_n_sol

        c1 = new_arr_time_next - sol_k['arrival_times'][next_n_sol] + waiting_time_next
        c1 = c1 / new_arr_time_next # normalize c1

        return c1

    def getC2(self, d, c1):
        """
        """
        return self.lambda_vrp*d['dist_from_depot'] - c1 + self.volw*d['vol']

    def getC3(self, sol_k, best_pos_ve_k):
        # c3(v) = c1(best_node(v)) + (volume of best_node(v))/(sum of volumes of all nodes in v)
        # best_pos_ve_k =  [<best_pos_k of that vehicle>]
        # best_pos_k = [<prev_n>,<next_n>,<c1>,<c2>,<node_id>]

        c1 = best_pos_ve_k[2]
        new_n_id = best_pos_ve_k[4]
        vol_new_n = self.delivery[str(new_n_id)]['vol']
        sum_prev_nodes = sol_k['init_vol'] - sol_k['vol_left']

        return self.beta1_c3*c1*self.env.conv_time_to_cost + \
             self.beta2_c3*(vol_new_n / (sum_prev_nodes + vol_new_n))*sol_k['cost']


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
            c2_d = self.getC2(self.delivery[str(d_id)], best_pos_all[d_id][2])

            #print(f"[DEBUG] c2_d: {c2_d}")

            # compare the cost c2 of the currently selected delivery "d_id"
            # with the optimum one. Update the optimum if better.
            if best_d[0] == None: # first check
                best_d = [d_id, c2_d]
            elif self.compareC2(c2_d, best_d[1]):
                best_d = [d_id, c2_d]

        return best_d[0] # return the id of the delivery with optimum c2

    def insertNode(self, sol_k, best_d_id, prev_n_sol, next_n_sol):
        """
        Insert a delivery in a vehicle's solution at a specific position with its arrival
        and waiting times and update the time info of the following deliveries if needed.
        """

        # Check if the insertion is being made in an empty vehicle:
        # in that case, insert the depot as first and last element of the 
        # path first
        if sol_k['n_nodes'] == 0:
            sol_k['path'].append(0)
            sol_k['arrival_times'].append(0)
            sol_k['waiting_times'].append(0)
            sol_k['path'].append(0)
            sol_k['arrival_times'].append(0)
            sol_k['waiting_times'].append(0)

        # DEBUG
        if sol_k['path'][prev_n_sol] == best_d_id or \
            sol_k['path'][next_n_sol] == best_d_id:
            print("DEBUG")

        # 1) Add the new delivery in the chosen place
        sol_k['path'].insert(prev_n_sol+1, int(best_d_id))
        # update the index of next_n
        next_n_sol += 1

        # 2) insert the new arrival and waiting times
        prev_n_id = sol_k['path'][prev_n_sol]
        dist_prev_d = self.distance_matrix[prev_n_id, best_d_id] # distance between prev_n and d
        arr_time_d = sol_k['arrival_times'][prev_n_sol] + \
            sol_k['waiting_times'][prev_n_sol] + \
                dist_prev_d
        waiting_time_d = max(0, self.delivery[str(best_d_id)]['time_window_min'] - arr_time_d)
        sol_k['arrival_times'].insert(prev_n_sol+1, arr_time_d)
        sol_k['waiting_times'].insert(prev_n_sol+1, waiting_time_d)


        # 3) update all the arrival & waiting times of the following deliveries
        next_n_id = sol_k['path'][next_n_sol]
        dist_d_next = self.distance_matrix[best_d_id, next_n_id] # distance between d and next_n
        new_arr_time_next = arr_time_d + waiting_time_d + dist_d_next

        additional_delay_flag = True
        while additional_delay_flag and next_n_sol < len(sol_k['path']):
            old_arr_time_next = sol_k['arrival_times'][next_n_sol]
            # Update the arrival time at next_n
            sol_k['arrival_times'][next_n_sol] = new_arr_time_next
            if next_n_sol != len(sol_k['path'])-1: # NOT the depot
                # DEBUG
                if next_n_id == 0:
                    print("DEBUG")
                delta_arr_time_next = new_arr_time_next - max(old_arr_time_next, self.delivery[str(next_n_id)]['time_window_min'])
                arr_time_relative = self.delivery[str(next_n_id)]['time_window_min']-new_arr_time_next
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
        sol_k['vol_left'] -= self.delivery[str(best_d_id)]['vol']
        # 5) increment the number of nodes in the vehicle
        sol_k['n_nodes'] += 1
        # 6) set the chosen_vrp of the delivery to True
        self.delivery[str(best_d_id)]['chosen_vrp'] = True

    def removeNode(self, sol_k, n_id, prev_n_sol, next_n_sol):
        # remove the delivery from the solution
        sol_k['path'].remove(n_id)
        sol_k['arrival_times'].pop(prev_n_sol+1)
        sol_k['waiting_times'].pop(prev_n_sol+1)
        next_n_sol -= 1
        # decrement the number of nodes in the vehicle
        sol_k['n_nodes'] -= 1

        if sol_k['n_nodes'] == 0: # removed the only node in the path
            sol_k['path'] = []
            sol_k['arrival_times'] = []
            sol_k['waiting_times'] = []
        else: 
            # update the arrival & waiting times of the following deliveries
            prev_n_id = sol_k['path'][prev_n_sol]
            next_n_id = sol_k['path'][next_n_sol]
            dist_prev_next = self.distance_matrix[prev_n_id,next_n_id]
            new_arr_time_next = sol_k['arrival_times'][prev_n_sol] + sol_k['waiting_times'][prev_n_sol] \
                + dist_prev_next

            update_flag = True
            while update_flag and next_n_sol < len(sol_k['path']):
                old_arr_time_next = sol_k['arrival_times'][next_n_sol]
                # Update the arrival time at next_n
                sol_k['arrival_times'][next_n_sol] = new_arr_time_next
                if next_n_sol != len(sol_k['path'])-1: # NOT the depot
                    # difference in the arrival time at the FOLLOWING node w.r.t. next_n
                    delta_arr_time_next = old_arr_time_next - max(self.delivery[str(next_n_id)]['time_window_min'], new_arr_time_next)
                    sol_k['waiting_times'][next_n_sol] = max(0, self.delivery[str(next_n_id)]['time_window_min'] - new_arr_time_next)
                    if old_arr_time_next > self.delivery[str(next_n_id)]['time_window_min']: 
                        # the following deliveries have to be updated
                        next_n_sol +=1
                        if next_n_sol < len(sol_k['path']):
                            next_n_id = sol_k['path'][next_n_sol]
                            new_arr_time_next = sol_k['arrival_times'][next_n_sol] - delta_arr_time_next
                    else:
                        update_flag = False
                else:
                    next_n_sol += 1

        # update the volume left in the vehicle
        sol_k['vol_left'] += self.delivery[str(n_id)]['vol']
        # set the chosen_vrp of the delivery to False
        self.delivery[str(n_id)]['chosen_vrp'] = False

    def restore_vehicles_order(self, sol):
        final_sol = [None]*len(sol)
        for k in range(len(sol)):
            final_sol[self.vehicles_order[k]] = sol[k]
        return final_sol

    def learn_and_save(self):
        self.learning_flag = True
        n = 4 # num of iterations to test each parameter
        # num of iterations used in the ALNS algorithm
        alns_N_max = 2000
        alns_N_IwI = 200

        
        # find a good vehicles permutation only during the first
        # time that learn_and_save() is called from the main.
        if not self.vehicles_dict:
            self.vehicles_dict = self.env.get_vehicles()
            initial_vehicles_dict = self.env.get_vehicles()
            vehicles_order = list(range(0, len(self.vehicles_dict)))
            # sort the vehicles based on their "appetibility", defined as:
            #   (1-cost_veh/sum_costs_vehicles) + vol_veh/sum_vols_vehicles
            sum_costs_vehicles = sum([v['cost'] for v in self.vehicles_dict])
            sum_vols_vehicles = sum([v['capacity'] for v in self.vehicles_dict])
            """ vehicles_order.sort(key=lambda x:\
                (1 - self.vehicles_dict[x]['cost']/sum_costs_vehicles) + \
                    self.vehicles_dict[x]['capacity']/sum_vols_vehicles, 
                reverse = True) """
            # test various vehicles_dict permutations 
            veh_all_orders = []
            ind = -1
            best_ind = None
            best_obj = None
            for i in range(2*n):
                veh_order_new = []
                D = vehicles_order.copy()
                for k in range(len(vehicles_order)):
                    y = np.random.uniform()
                    v = D[int(np.power(y,self.veh_p)*(len(D)-1))]
                    veh_order_new.append(v)
                    D.remove(v)
                # check if the permutation was already generated before
                if not veh_order_new in veh_all_orders:
                    # if not, insert it and evaluate the obj function with this permutation
                    veh_all_orders.append(veh_order_new)
                    
                    ind += 1
                    # translate the permutation into a valid vehicles_dict
                    new_vehicles_dict = []
                    for v in veh_order_new:
                        new_vehicles_dict.append(initial_vehicles_dict[v])
                    # temporarily save the new permutation as the best one. It is needed 
                    # to restore the original vehicles scheme when calling evaluate_VRP
                    self.vehicles_order = veh_order_new                    
                    self.vehicles_dict = new_vehicles_dict
                    
                    #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
                    #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
                    VRP_solution = self.compute_VRP(self.env.get_delivery(), new_vehicles_dict, alns_N_max, alns_N_IwI)
                    obj = self.env.evaluate_VRP(VRP_solution)
                    # DEBUG
                    print(f"Perm: {veh_order_new}, obj: {obj}")
                    if not best_obj:
                        best_obj = obj
                        best_ind = ind
                    elif obj < best_obj:
                        best_obj = obj
                        best_ind = ind
                #self.veh_p *= 1.5 # decrease the randomness of the vehicle permutation choice
            # adopt the best permutation found
            self.vehicles_order = veh_all_orders[best_ind]
            best_vehicles_dict = []
            # DEBUG
            print()
            print(f"[DEBUG] vehicles order: {veh_all_orders[best_ind]}")
            for v in veh_all_orders[best_ind]:
                best_vehicles_dict.append(initial_vehicles_dict[v])
            self.vehicles_dict = best_vehicles_dict


        # test volw
        volw_rnd = np.random.uniform(0.5, 3, n-1)
        volw_rnd = np.insert(volw_rnd, 0, self.volw)
        for i in range(n): 
            self.volw = volw_rnd[i]
            #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
            #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
            VRP_solution = self.compute_VRP(self.env.get_delivery(), self.env.get_vehicles(), alns_N_max, alns_N_IwI)
            obj = self.env.evaluate_VRP(VRP_solution)
            self.data_improv_volw['volw'].append(volw_rnd[i])
            self.data_improv_volw['obj'].append(obj)

        # test beta1_c3
        beta1_c3_rnd = np.random.uniform(0.5,0.95,n-1)
        beta1_c3_rnd = np.insert(beta1_c3_rnd, 0, self.beta1_c3)
        for i in range(n): 
            self.beta1_c3 = beta1_c3_rnd[i]
            self.beta2_c3 = 1-self.beta1_c3
            #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
            #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
            VRP_solution = self.compute_VRP(self.env.get_delivery(), self.env.get_vehicles(), alns_N_max, alns_N_IwI)
            obj = self.env.evaluate_VRP(VRP_solution)
            self.data_improv_beta1_c3['beta1_c3'].append(beta1_c3_rnd[i])
            self.data_improv_beta1_c3['obj'].append(obj)
        
        # test alns_eps
        alns_eps_rnd = np.random.uniform(0.9,0.99998,n-1)
        alns_eps_rnd = np.insert(alns_eps_rnd, 0, self.alns_eps)
        for i in range(n): 
            self.alns_eps = alns_eps_rnd[i]
            #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
            #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
            VRP_solution = self.compute_VRP(self.env.get_delivery(), self.env.get_vehicles(), alns_N_max, alns_N_IwI)
            obj = self.env.evaluate_VRP(VRP_solution)
            self.data_improv_alns_eps['alns_eps'].append(alns_eps_rnd[i])
            self.data_improv_alns_eps['obj'].append(obj)

        # test alns_p
        alns_p_rnd = np.random.uniform(1, 10, n-1)
        alns_p_rnd = np.insert(alns_p_rnd, 0, self.alns_p)
        for i in range(n): 
            self.alns_p = alns_p_rnd[i]
            #id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
            #remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
            VRP_solution = self.compute_VRP(self.env.get_delivery(), self.env.get_vehicles(), alns_N_max, alns_N_IwI)
            obj = self.env.evaluate_VRP(VRP_solution)
            self.data_improv_alns_p['alns_p'].append(alns_p_rnd[i])
            self.data_improv_alns_p['obj'].append(obj)

        self.learning_flag = False
    
    def start_test(self):
        # fix volw
        volw_pos_min = self.data_improv_volw['obj'].index(min(self.data_improv_volw['obj']))
        self.volw = self.data_improv_volw['volw'][volw_pos_min]
        print()
        print(f"[DEBUG] CHOSEN VOLW: {self.volw}")
        
        # fix beta1_c3 and beta2_c3
        beta1_pos_min = self.data_improv_beta1_c3['obj'].index(min(self.data_improv_beta1_c3['obj']))
        self.beta1_c3 = self.data_improv_beta1_c3['beta1_c3'][beta1_pos_min]
        self.beta2_c3 = 1-self.beta1_c3
        print(f"[DEBUG] CHOSEN beta1_c3: {self.beta1_c3}") 

        # fix alns_eps
        alns_eps_pos_min = self.data_improv_alns_eps['obj'].index(min(self.data_improv_alns_eps['obj']))
        self.alns_eps = self.data_improv_alns_eps['alns_eps'][alns_eps_pos_min]
        print(f"[DEBUG] CHOSEN alns_eps: {self.alns_eps}")

        # fix alns_p
        alns_p_pos_min = self.data_improv_alns_p['obj'].index(min(self.data_improv_alns_p['obj']))
        self.alns_p = self.data_improv_alns_p['alns_p'][alns_p_pos_min]
        print(f"[DEBUG] CHOSEN alns_p: {self.alns_p}")

        print()