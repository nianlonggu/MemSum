import os
from glob import glob
import torch
import torch.nn as nn
import json
from tqdm import tqdm



def load_model( model_folder ):
    ckpt_list =  glob( model_folder + "/*.pt" )
    if len( ckpt_list ) >0:
        ckpt_list.sort( key = os.path.getmtime )
        ckpt_name = ckpt_list[-1]
        if torch.cuda.is_available():
            ckpt = torch.load( ckpt_name )
        else:
            ckpt = torch.load( ckpt_name,  map_location=torch.device('cpu') )
    else:
        ckpt = None
    return ckpt

def save_model(  module_dicts ,save_name , max_to_keep = 0, overwrite = True ):
    folder_path = os.path.dirname( os.path.abspath( save_name )  )
    if not os.path.exists( folder_path  ):
        os.makedirs( folder_path )

    state_dicts = {}
    for key in module_dicts.keys():
        if isinstance( module_dicts[key], nn.DataParallel ):
            state_dicts[key] = module_dicts[key].module.state_dict()
        elif isinstance( module_dicts[key], nn.Module ):
            state_dicts[key] = module_dicts[key].state_dict()
        else:
            state_dicts[key] = module_dicts[key]

    if os.path.exists( save_name ):
        if overwrite:
            os.remove( save_name )
            torch.save( state_dicts, save_name )
        else:
            print("Warning: checkpoint file already exists!")
            return
    else:
        torch.save( state_dicts, save_name )

    if max_to_keep > 0:
        pt_file_list = glob(folder_path+"/*.pt")
        pt_file_list.sort( key= lambda x: os.path.getmtime(x) )
        for idx in range( len( pt_file_list ) - max_to_keep ):
            os.remove( pt_file_list[idx]  )


def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups ]

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])