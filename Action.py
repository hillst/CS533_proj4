#!/usr/bin/env python

class Action:
    #state x transition probability
    name_ = ""
    index_ = -1
    def __init__(self, name, index_):
        self.name_ = name   
        self.index_ = index

    def get_name(self):
        return self.name_

    def get_index(self):
        return self.index_


