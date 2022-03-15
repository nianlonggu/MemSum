import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-log_file_name" )
parser.add_argument("-top_N", type = int, default = 1 )

args = parser.parse_args()

batch_matcher = re.compile("^\[current_batch:\s+([0-9]+)\]") 
score_matcher = re.compile("val:\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)")

lines = open( args.log_file_name,"r" ).readlines()

batch_list = []
raw_score_list = []
score_list = []

for pos, line in enumerate(lines):
    if line.startswith("Starting validation"):
        if pos >0 and pos < len(lines)-1:
            batch = batch_matcher.findall( lines[pos-1] )
            score = score_matcher.findall( lines[pos+1] )

            if len(batch) >0 and len(score)>0:
                raw_score_list.append( score[0])
                batch = int(batch[0])
                score = np.mean(list(map( float, score[0] )))
                batch_list.append( batch )
                score_list.append( score )

#max_score_position = np.argmax(score_list)
top_N_poses = np.argsort( - np.array( score_list  )  )[:args.top_N]

for pos in top_N_poses:
    print("batch: %d"%(batch_list[pos]), "scores:", raw_score_list[pos])
