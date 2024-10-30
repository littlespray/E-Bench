import yaml

import torch
import torch.nn as nn

from .tradition import DOVER
from .fidelity import DoubleStreamModel
from .text_alignment import VideoTextAlignmentModel


class EvalEditModel(nn.Module):
    def __init__(self):
        super().__init__()
        # load cfg
        dover_config = 'configs/dover.yaml'
        doublestream_config = 'configs/doublestream.yaml'
        text_config = 'configs/text.yaml'
        
        with open(dover_config, "r") as f:
            dover_opt = yaml.safe_load(f)
        with open(doublestream_config, "r") as f:
            doublestream_opt = yaml.safe_load(f)
        with open(text_config, "r") as f:
            text_opt = yaml.safe_load(f)
        
        #dover_ckpt=[(**dover_opt['model']['test_load_path'])

        # build model TODO
        self.traditional_branch = DOVER(**dover_opt['model']['args']).eval()
        self.fidelity_branch = DoubleStreamModel(**doublestream_opt['model']['args']).eval()
        self.text_branch = VideoTextAlignmentModel(**text_opt['model']['args']).eval()
        ckpts =[dover_opt['test_load_path'][0],doublestream_opt['test_load_path'][0],text_opt['test_load_path'][0]]
        
        # load_weight TODO
        self.load_ckpt(ckpts)

    # TODO
    def load_ckpt(self, ckpt_folder):
        self.traditional_branch.load_state_dict(torch.load(ckpt_folder[0],map_location='cpu')['state_dict'])
        self.fidelity_branch.load_state_dict(torch.load(ckpt_folder[1],map_location='cpu')['state_dict'],strict=False)
        self.text_branch.load_state_dict(torch.load(ckpt_folder[2],map_location='cpu')['state_dict'],strict=False)
    # TODO
    def forward(self, src_video, edit_video, prompt):
        traditional_score = self.traditional_branch(edit_video,reduce_scores=True)
        fidelity_score = self.fidelity_branch(src_video, edit_video)
        text_score = self.text_branch(edit_video,prompts=prompt)
        # the weight of each score is pre-computed within each branch
        return (traditional_score + fidelity_score[0] + text_score[0]).item()



if __name__ == "__main__":
    eval_model=EvalEditModel()
