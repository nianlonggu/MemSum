from utils import greedy_extract
import json
from tqdm import tqdm
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-input_corpus_file_name" )
    parser.add_argument("-output_corpus_file_name" )
    parser.add_argument("-beamsearch_size", type = int, default = 2)
    parser.add_argument("-start", type =int, default = 0 )
    parser.add_argument("-size", type =int, default = 0 )

    args = parser.parse_args()

    output_corpus_file_name = args.output_corpus_file_name+"_%d"%(args.start)

    with open( output_corpus_file_name, "w" ) as fw:
        count = 0
        with open(args.input_corpus_file_name,"r") as f:
            for line in tqdm(f):
                if count < args.start:
                    count +=1
                    continue
                if args.size>0 and count >= args.start + args.size:
                    break   
                    
                data = json.loads(line)
                
                try:
                    high_rouge_episodes = greedy_extract( data["text"], data["summary"], beamsearch_size = args.beamsearch_size )
                    indices_list = []
                    score_list  = []

                    for indices, score in high_rouge_episodes:
                        indices_list.append( indices )
                        score_list.append(score)

                    data["indices"] = indices_list
                    data["score"] = score_list
                        
                    if not( len(data["indices"]) == 0 or len(data["indices"]) != len(data["score"]) or len(data["text"]) == 0 or len(data["summary"]) == 0 ):
                        fw.write( json.dumps( { 
                                            "text": data["text"],
                                            "summary": data["summary"],
                                            "indices": data["indices"],
                                            "score" : data["score"]
                                        }  ) + "\n" )
                except:
                    print("parsing error! skip.")

                count +=1

    print("finished!")
