#!/usr/bin/env python

class State:
    name_ = ""
    index_ = ""
    terminal_ = False
    def __init__(self, name, index, terminal):
        self.name_ = name
        self.index_ = index
        self.terminal_ = terminal

    def get_name(self):
        return self.name_
    
    def get_index(self):
        return self.index_

    def is_terminal(self):
        return self.terminal_
        
