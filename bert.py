from pathlib import Path
import torch
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import time

# load model and tokenizer
feature = "sequence-classification"

model_id="textattack/bert-base-uncased-yelp-polarity"
save_name="bert-base"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)

print(model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "This is a sample sentence."
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
print(encoded_input)

start_time = time.time()
outputs = model(**encoded_input)
end_time = time.time()
processing_time = end_time - start_time

# print(outputs)
print(f"Output shape:{outputs.logits.shape} Processing time for a query: {processing_time:.4f} seconds")

