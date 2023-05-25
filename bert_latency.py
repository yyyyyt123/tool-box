from transformers import AutoTokenizer, BertForMaskedLM
import torch
import time
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased").cuda()

# HyperParameters
bs=2
sl=32
FP16 = True

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
        logits = model(**inputs).logits
    
    # test for 10 times
    for i in range(10):
      for k,v in inputs.items():
        inputs[k]=inputs[k].cuda()
        #print(inputs[k].shape)
      torch.cuda.synchronize()
      with torch.no_grad():
        starter.record()
        logits = model(**inputs).logits
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        times.append(curr_time)
    print("bs:{:>4d} seq_length:{:>4d} mean latency:{:>5f}".format(bs, sl, np.mean(times)))

# test begin
for bs in [1,2,4,8,16]:
  for sl in [32,64,128,256]:
    inference_measure(bs, sl)
