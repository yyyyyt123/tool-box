#!/bin/bash -x

python bert_latency.py > out.txt

python bert_load_time.py >> out.txt