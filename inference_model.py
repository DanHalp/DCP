# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

from easydict import EasyDict as edict



class DCP_MODEL():
    
    def __init__(self, args) -> None:
        self._model, self._cfg  = self.make_model(args)
        
    
    def make_model(self, args=edict()):
       

    @property
    def model(self):
        return self._model
    
    @property
    def cfg(self):
        return self._cfg

    def test(self):
        self.model.test()
    
