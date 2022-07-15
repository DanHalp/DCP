# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from SpareNet.configs.base_config import cfg as sparenet_cfg
from easydict import EasyDict as edict
from pathlib import Path
import torch



class DCP_MODEL():
    
    def __init__(self, args) -> None:
        self._model, self._cfg  = self.make_model(args)
        
    
    def make_model(self, args=edict()):
        
        test_dataset_loader = DATASET_LOADER_MAPPING[cfg.DATASET.test_dataset](cfg)
        val_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(DatasetSubset.TEST),
            batch_size=1,
            num_workers=cfg.CONST.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )
       

    @property
    def model(self):
        return self._model
    
    @property
    def cfg(self):
        return self._cfg

    def test(self):
        self.model.test()
    
