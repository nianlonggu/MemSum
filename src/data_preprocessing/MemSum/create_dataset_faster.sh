#!/bin/bash

for (( start=$1; start<$2; start += $3))
    do
        python create_dataset_faster.py -input_corpus_file_name $4  -output_corpus_file_name $5 -beamsearch_size $6 -max_num_extracted_sentences $7 -max_num_extractions $8 -start $start -size $3 -epsilon $9 -truncation ${10} &
    done