from transformers import AutoTokenizer, BertForMaskedLM, BertModel
import torch
import argparse
import time
import numpy as np


parser = argparse.ArgumentParser(description='BERT inference latency measurement')
parser.add_argument('--fp16', action='store_true', help='use fp16 for inference')
args = parser.parse_args()
FP16 = args.fp16

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BertModel.from_pretrained(model_id).cuda()

# HyperParameters
print(f"using model_id :{model_id}")
print("Use", "FP16" if FP16 is True else "FP32", "in inference")
# model
model = model.half() if FP16 is True else model

def inference_measure(bs, sl):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times=[]
    text=["The capital of France is [MASK]"] * bs
    inputs = tokenizer.batch_encode_plus(text, padding="max_length",max_length=sl,return_tensors="pt")
    # warm up for 3 times
    for i in range(3):
      for k,v in inputs.items():
        inputs[k]=inputs[k].cuda()
      with torch.no_grad():
        logits = model(**inputs)
    
    # test for 10 times
    for i in range(10):
      for k,v in inputs.items():
        inputs[k]=inputs[k].cuda()
        #print(inputs[k].shape)
      torch.cuda.synchronize()
      with torch.no_grad():
        starter.record()
        logits = model(**inputs)
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        times.append(curr_time)
    print("bs:{:>4d} seq_length:{:>4d} mean latency:{:>5f}".format(bs, sl, np.mean(times)))

# test begin
for bs in [1,2,4,8,16,32,64]:
  for sl in [32,64,128,256,512]:
    inference_measure(bs, sl)
