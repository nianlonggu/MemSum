import json
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-folder" )
parser.add_argument("-prefix" )
parser.add_argument("-save_name" )

args = parser.parse_args()

if __name__ == "__main__":
    flist = [ args.folder + "/"+ fname for fname in os.listdir(args.folder) if fname.startswith( args.prefix ) ]
    flist.sort( key = lambda x: int( x.split("_")[-1] ) )

    fw = open( args.folder + "/" + args.save_name ,"w")

    for fname in flist:
        print(fname)
        with open( fname,"r" ) as f:
            for line in f:
                fw.write(line)

    fw.close()