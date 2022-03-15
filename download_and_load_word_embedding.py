import numpy as np
import pickle
import torch

import os
os.chdir("model/glove/")
os.system("wget https://nlp.stanford.edu/data/glove.6B.zip")
os.system("unzip glove.6B.zip")

embed = []
with open("glove.6B.200d.txt","r") as f:
    for line in f:
        embed.append( line.strip().split()[1:] )
embed = np.asarray( embed).astype(np.float32)
embed = np.concatenate( [ np.zeros( (3, 200) ), embed ], axis = 0 ).astype(np.float32)

with open("unigram_embeddings_200dim.pkl","wb") as f:
    pickle.dump( embed, f, -1 )
    
os.chdir("../")
all_model_files = []
for root, dirs, files in os.walk("./"):
    for fname in files:
        if fname.endswith(".pt"):
            all_model_files.append(root+"/"+fname)

for model_name in all_model_files:
    model_ckpt = torch.load( model_name ,  map_location = "cpu"  )
    model_ckpt["local_sentence_encoder"]["word_embedding"] = torch.from_numpy( embed )
    torch.save( model_ckpt, model_name )

print("All model loaded!")