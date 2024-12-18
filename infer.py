import torch
import torch.nn as nn

from models import EvalEditModel
from preprocess import Processor
import yaml
import argparse

#fixed seed
seed_n = 42
print('seed is ' + str(seed_n))
torch.manual_seed(seed_n)

device='cuda'
class EBenchModel(nn.Module):
    def __init__(self):
        super().__init__()
        dover_config = 'configs/dover.yaml'
        doublestream_config = 'configs/doublestream.yaml'
        text_config = 'configs/text.yaml'

        with open(dover_config, "r") as f:
            dover_opt = yaml.safe_load(f)
        with open(doublestream_config, "r") as f:
            doublestream_opt = yaml.safe_load(f)
        with open(text_config, "r") as f:
            text_opt = yaml.safe_load(f)
        self.model = EvalEditModel().cuda()
        self.traditional_processor=Processor(dover_opt['data']['videoQA']['args'])
        self.text_pocessor=Processor(text_opt['data']['videoQA']['args'])
        self.doublestream_processor=Processor(doublestream_opt['data']['videoQA']['args'])

    
    def read_data(self, path):
        traditional_data=self.traditional_processor.preprocess(path)
        text_data=self.text_pocessor.preprocess(path)
        doublestream_data = self.doublestream_processor.preprocess(path)
        data={}
        for branch_data in[traditional_data,text_data,doublestream_data]:
            for key in branch_data.keys():
                data[key]=branch_data[key]
        return data
    
    
    @torch.no_grad()
    def evaluate(self, prompt, src_path, dst_path):
        src_video = self.read_data(src_path)
        dst_video = self.read_data(dst_path)
        result = self.model(src_video, dst_video, prompt)
        return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process video files with EBenchModel.')


    parser.add_argument('--single_test', action='store_true', help='Run a single test with specified paths and prompt.')
    parser.add_argument('--src_path', type=str, help='Source video path for single test.')
    parser.add_argument('--dst_path', type=str, help='Destination video path for single test.')
    parser.add_argument('--prompt', type=str, help='Prompt for single test.')
    parser.add_argument('--data_path', type=str, help='Data path for batch processing.')
    parser.add_argument('--label_path', type=str, help='Label path for batch processing.')


    args = parser.parse_args()


    if args.single_test:
        if args.src_path and args.dst_path and args.prompt:
            src_path = args.src_path
            dst_path = args.dst_path
            prompt = args.prompt
            ebench = EBenchModel()
            result = ebench.evaluate(prompt, src_path, dst_path)
            print(f"The result is {result}")
        else:
            print("Error: For single test, --src_path, --dst_path, and --prompt must be provided.")
    else:
        if args.data_path and args.label_path:
            data_path = args.data_path
            label_path = args.label_path
            src=[]
            dst=[]
            prompts=[]
            with open(label_path,'r') as file:
                for line in file:
                    video_name,_,prompt=line.split('|')
                    src+=[data_path+"src/"+video_name]
                    dst += [data_path + "edited/" + video_name]
                    prompts+=[prompt]
            ebench = EBenchModel()
            results=[]
            for src_path,dst_path,prompt in zip(src,dst,prompts):
                result = ebench.evaluate(prompt, src_path, dst_path)
                results+=[result]
                print(len(results))
            with open("label.txt","w") as file:
                for src_path,result in zip(src,results):
                    file.write(f"{src_path.split('/')[-1]},{result}\n")
        else:
            print("Error: For batch test, --data_path, --label_path must be provided.")
