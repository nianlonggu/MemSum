import threading
import subprocess
import numpy as np
from glob import glob
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_corpus_file_name" )
    parser.add_argument("-output_corpus_file_name" )
    parser.add_argument("-beamsearch_size", type = int, default = 2)
    parser.add_argument("-n_processes", type =int, default = 1 )

    args = parser.parse_args()
    
    total_num = 0
    for line in open(args.input_corpus_file_name):
        total_num +=1    
    
    size_per_process = int( np.ceil( total_num / args.n_processes ) )
    
    threads = []
    for start in range( 0, total_num, size_per_process ):
        t = threading.Thread( target = subprocess.run,
                          args = (
                           [ "python", "get_high_rouge_episodes_sp.py",
                            "-input_corpus_file_name", args.input_corpus_file_name,
                            "-output_corpus_file_name", args.output_corpus_file_name,
                            "-beamsearch_size", str(args.beamsearch_size),
                            "-start", str( start ),
                            "-size", str( size_per_process )
                           ],
                          )
                        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    flist = glob( args.output_corpus_file_name + "_*" )
    flist.sort(key = lambda x:int( x.split("_")[-1] ))
    
    fw = open(args.output_corpus_file_name,"w")
    
    for fname in flist:
        with open(fname) as f:
            for line in f:
                fw.write(line)
        os.remove( fname )
    fw.close()
    