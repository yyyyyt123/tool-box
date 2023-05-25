import time
import torch
import numpy as np
from transformers import BertModel

# Load the BERT model into CPU memory
model_id = "bert-base-uncased"
model = BertModel.from_pretrained(model_id).to('cpu')

# Get the parameters of the first encoder layer
print(f"using model_id :{model_id}")

# Store the layer parameters to disk
bert_path = 'Bert_model.pth'                 # whole model
embedding_layer_path = 'Embedding_layer.pth' # embedding layer
encoder_layer_path = 'Encoder_layer.pth'     # 12 layer encoder
attention_layer_path = 'Attention_layer.pth' # Attention 

bert_model = model
embedding_layer_model = model.embeddings
encoder_layer_model = model.encoder
attention_layer_model = model.encoder.layer[0].attention

torch.save(model.state_dict(), bert_path)
torch.save(model.embeddings.state_dict(), embedding_layer_path)
torch.save(model.encoder.state_dict(), encoder_layer_path)
torch.save(model.encoder.layer[0].attention.state_dict(), attention_layer_path)

def DISK2CPU_time_measure(file_path, model, warm_up_times=3, measure_times=10):
    assert file_path is not None
    # Warm-up rounds
    for _ in range(warm_up_times):
        _model = torch.load(file_path)
        # del _model

    # Measure the time to load the first layer from disk 
    loading_times = []
    for _ in range(measure_times):
        start_time = time.time()
        model.load_state_dict(torch.load(file_path, map_location='cpu'))
        end_time = time.time()
        loading_time = end_time - start_time
        loading_times.append(loading_time)

    # Calculate the average loading time (excluding warm-up rounds)
    average_loading_time = np.mean(loading_times)
    num_params = sum(p.numel() for p in model.parameters())
    print("DISK2CPU Loading:{:>20s} --- # Params: {:>10d} --- takes :{:>3f} seconds".format(file_path[:-4], num_params,average_loading_time))

def CPU2GPU_time_measure(file_path, model, warm_up_times=3, measure_times=10):
    assert torch.cuda.is_available() is True

    device = torch.device("cuda")
    # Warm-up rounds
    for _ in range(warm_up_times):
        _model = model.to(device)
        # del _model

    # Measure the time to load the first layer from disk 
    loading_times = []
    for _ in range(measure_times):
        start_time = time.time()
        # model.load_state_dict(torch.load(file_path, map_location='cpu'))
        model = model.to(device)
        torch.cuda.synchronize()
        end_time = time.time()
        loading_time = end_time - start_time
        loading_times.append(loading_time)

    # Calculate the average loading time (excluding warm-up rounds)
    average_loading_time = np.mean(loading_times)
    num_params = sum(p.numel() for p in model.parameters())
    print("CPU2GPU Loading: {:>20s} --- # Params: {:>10d} --- takes :{:>3f} seconds".format(file_path, num_params,average_loading_time))

DISK2CPU_time_measure(bert_path, bert_model)
DISK2CPU_time_measure(embedding_layer_path, embedding_layer_model)
DISK2CPU_time_measure(encoder_layer_path, encoder_layer_model)
DISK2CPU_time_measure(attention_layer_path, attention_layer_model)

CPU2GPU_time_measure(bert_path[:-4], bert_model)
CPU2GPU_time_measure(embedding_layer_path[:-4], embedding_layer_model)
CPU2GPU_time_measure(encoder_layer_path[:-4], encoder_layer_model)
CPU2GPU_time_measure(attention_layer_path[:-4], attention_layer_model)
