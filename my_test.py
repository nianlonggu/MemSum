from my_summarizers import ExtractiveSummarizer_MemSum_Final, ExtractiveSummarizer_MemSum_wo_history, ExtractiveSummarizer_MemSum_with_stop_sentence
from pyrouge import Rouge155
import json
import numpy as np
import os
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model_type" )
parser.add_argument("-summarizer_model_path" )
parser.add_argument("-vocabulary_path" )
parser.add_argument("-corpus_path" )
parser.add_argument("-gpu", type =int )
parser.add_argument("-max_extracted_sentences_per_document", type =int )
parser.add_argument("-p_stop_thres", type =float )
parser.add_argument("-output_file" )
parser.add_argument("-max_count", type = int, default = None)

parser.add_argument("-use_fast_rouge", type = int, default = 1)
parser.add_argument("-system_dir", default = None )
parser.add_argument("-model_dir", default = None )


parser.add_argument("-N_enc_l", type =int, default = 2 )
parser.add_argument("-N_enc_g", type =int, default = 2 )
parser.add_argument("-N_dec", type =int, default = 3 )
parser.add_argument("-embed_dim", type =int, default = 200 )
parser.add_argument("-ngram_blocking", default = "False" )
parser.add_argument("-ngram", type =int, default = 4 )
parser.add_argument("-max_doc_len", type = int, default = 500)
parser.add_argument("-max_seq_len", type = int, default = 100)


args = parser.parse_args()

ngram_blocking = args.ngram_blocking.lower() == "true"

results_dir_path = os.path.dirname( args.output_file )
if not os.path.exists( results_dir_path ):
    try:
        os.makedirs( results_dir_path )
    except:
        pass


if args.model_type == "MemSum_Final":
    base_extractor = ExtractiveSummarizer_MemSum_Final( args.summarizer_model_path,
                                      args.vocabulary_path, 
                                      gpu = args.gpu,
                                      N_enc_l = args.N_enc_l,
                                      N_enc_g = args.N_enc_g,
                                      N_dec = args.N_dec,
                                      embed_dim = args.embed_dim,
                                      max_doc_len  = args.max_doc_len,
                                      max_seq_len = args.max_seq_len
                                    )

elif args.model_type == "MemSum_wo_history":
    base_extractor = ExtractiveSummarizer_MemSum_wo_history( args.summarizer_model_path,
                                      args.vocabulary_path, 
                                      gpu = args.gpu,
                                      N_enc_l = args.N_enc_l,
                                      N_enc_g = args.N_enc_g,
                                      N_dec = args.N_dec,
                                      embed_dim = args.embed_dim,
                                      max_doc_len  = args.max_doc_len,
                                      max_seq_len = args.max_seq_len
                                    )
elif args.model_type == "MemSum_with_stop_sentence":
    base_extractor = ExtractiveSummarizer_MemSum_with_stop_sentence( args.summarizer_model_path,
                                      args.vocabulary_path, 
                                      gpu = args.gpu,
                                      N_enc_l = args.N_enc_l,
                                      N_enc_g = args.N_enc_g,
                                      N_dec = args.N_dec,
                                      embed_dim = args.embed_dim,
                                      max_doc_len  = args.max_doc_len,
                                      max_seq_len = args.max_seq_len
                                    )


print("Start Computation ...")

if args.use_fast_rouge:
    from rouge_score import rouge_scorer
    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

    count = 0
    # extracted_len_list = []
    # original_len_list = []
    num_extracted_sentences_list = []
    extraction_time_list = []
    scores = []

    with open(args.corpus_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if " ".join( data["summary"]).strip() == "":
                continue
        
            tic = time.time()
            if args.model_type == "Oracle":
                extracted_sen = [  data["text"][idx] for idx in data["indices"][0] ]
            elif args.model_type.startswith("Lead_"):
                lead_n = int( args.model_type.split("_")[-1] )
                extracted_sen = data["text"][:lead_n]
            else:
                extracted_sen = base_extractor.extract( [ data["text"] ], args.p_stop_thres , ngram_blocking, args.ngram , max_extracted_sentences_per_document = args.max_extracted_sentences_per_document )
                extracted_sen = extracted_sen[0]
            tac = time.time()
            extraction_time_list.append(tac - tic)
            num_extracted_sentences_list.append( len( extracted_sen ) )

            score = rouge_cal.score( "\n".join( data["summary"] ) , "\n".join( extracted_sen )  )
            scores.append( ( score["rouge1"] ,score["rouge2"],score["rougeLsum"]     ) )


            count +=1
            if args.max_count is not None and count >= args.max_count:
                break

    rouge1_scores, rouge2_scores, rougel_scores = list(zip(*scores))
    res = (       np.asarray( rouge1_scores ).mean(axis = 0).tolist() ,
                  np.asarray( rouge2_scores ).mean(axis = 0).tolist() ,
                  np.asarray( rougel_scores ).mean(axis = 0).tolist() ,
                  )
    with open(args.output_file, "a") as f:
        info = "p_stop_thres: %.4f, avg. # sentences: %.2f ± %.2f, avg. extraction time: %.2f ± %.2f ms, R-1 (p,r,f1): %.4f, %.4f, %.4f R-2: %.4f, %.4f, %.4f \t R-L: %.4f, %.4f, %.4f\n" % \
                (args.p_stop_thres, np.mean( num_extracted_sentences_list ), np.std( num_extracted_sentences_list ) , np.mean( extraction_time_list ) * 1000, np.std( extraction_time_list ) * 1000   , res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2], res[2][0], res[2][1], res[2][2],  )

        f.write(info)
    print(info)

else:

    from pyrouge import Rouge155
    if not os.path.exists( args.system_dir ):
        try:
            os.makedirs( args.system_dir  )
        except:
            pass
    if not os.path.exists( args.model_dir ):
        try:
            os.makedirs( args.model_dir  )
        except:
            pass
    os.system("rm -r %s/*"%( args.system_dir))
    os.system("rm -r %s/*"%(  args.model_dir))


    r = Rouge155()
    r.system_dir = args.system_dir
    r.model_dir = args.model_dir
    r.system_filename_pattern = 'text.(\d+).txt'
    r.model_filename_pattern = 'text.[A-Z].#ID#.txt'

    count = 0
    # extracted_len_list = []
    # original_len_list = []
    num_extracted_sentences_list = []
    extraction_time_list = []

    with open(args.corpus_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if " ".join( data["summary"]).strip() == "":
                continue

            # original_len_list+=[len(idx) for idx in data["indices"] ]
            with open("%s/text.%d.txt"%(args.system_dir, count), "w") as f:
                tic = time.time()
                if args.model_type == "Oracle":
                    extracted_sen = [  data["text"][idx] for idx in data["indices"][0] ]
                elif args.model_type.startswith("Lead_"):
                    lead_n = int( args.model_type.split("_")[-1] )
                    extracted_sen = data["text"][:lead_n]
                else:
                    extracted_sen = base_extractor.extract( [ data["text"] ], args.p_stop_thres , ngram_blocking, args.ngram , max_extracted_sentences_per_document = args.max_extracted_sentences_per_document )
                    extracted_sen = extracted_sen[0]
                tac = time.time()
                extraction_time_list.append(tac - tic)
                num_extracted_sentences_list.append( len( extracted_sen ) )
                f.write( "\n".join( extracted_sen )  ) 
            # extracted_len_list.append(len(extracted_sen))
            with open( "%s/text.A.%d.txt"%(args.model_dir, count), "w" ) as f:
                f.write( "\n".join( data["summary"] ) )
            count +=1
            if args.max_count is not None and count >= args.max_count:
                break

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)

    with open(args.output_file, "a") as f:
        info = "p_stop_thres: %.4f, avg. # sentences: %.2f ± %.2f, avg. extraction time: %.2f ± %.2f ms, %s" % \
                (args.p_stop_thres, np.mean( num_extracted_sentences_list ), np.std( num_extracted_sentences_list ) , np.mean( extraction_time_list ) * 1000, np.std( extraction_time_list ) * 1000 , json.dumps(output_dict) + "\n" )

        f.write(info)
    print(info)
