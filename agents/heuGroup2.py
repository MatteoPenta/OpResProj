# -*- coding: utf-8 -*-
import time
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
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        # evaluate the distance of each point from the ORIGIN
        distance_matrix = spatial.distance_matrix([[0,0]], points)

        for i in range(len(distance_matrix[0, :])):
            # evaluate the score of the delivery
            self.delivery[i]['score'] = self.deliv_crowds_weights['a']*(1-self.delivery[i]['p_failed']) + \
                self.deliv_crowds_weights['b']*distance_matrix[0,i]
            print(f"[DEBUG] Score of node {self.delivery[i]['id']}: {self.delivery[i]['score']}")
            print(f"        Distance of node {self.delivery[i]['id']}: {distance_matrix[0,i]}")

        # 2) evaluate the threshold based on self.quantile
        threshold = np.quantile([dlv['score'] for dlv in self.delivery], self.quantile)
        print(f"Threshold: {threshold}")
        threshold_dist = np.quantile(distance_matrix[0, :], self.quantile)
        print(f"Disance threshold: {threshold_dist}")
        
        # 3) select the deliveries with score above threshold
        id_to_crowdship = []
        for i in range(len(self.delivery)):
            if self.delivery[i]['score'] > threshold:
                id_to_crowdship.append(i)
        
        return id_to_crowdship 

    def compute_VRP(self, delivery_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):
        ris = []
        print("Ciao")
        return ris

    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
