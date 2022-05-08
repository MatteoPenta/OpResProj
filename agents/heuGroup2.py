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

    def compute_delivery_to_crowdship(self, deliveries):
        # 1) evaluate the score for all deliveries
        if len(deliveries) == 0:
            return []
        points = []
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        distance_matrix = spatial.distance_matrix(points, points)

        
        # 2) evaluate the threshold based on self.quantile

        # 3) select the deliveries with score above threshold

    def compute_VRP(self, delivery_to_do, vehicles):
        ris = []
        return ris

    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
