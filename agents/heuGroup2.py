# -*- coding: utf-8 -*-
import time
from agents import *


class HeuGroup2(Agent):

    def __init__(self, env):
        self.name = "HeuGroup2"
        self.env = env

    def compute_delivery_to_crowdship(self, deliveries):
        return [i + 1 for i in range(len(deliveries))]

    def compute_VRP(self, delivery_to_do, vehicles):
        ris = []
        print("Ciao")
        return ris

    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
