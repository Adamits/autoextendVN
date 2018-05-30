import torch
import numpy as np
import gensim
import pickle

import verbnet_extractor
from autoencoder import AutoEncoder
from loss import autoExtend_loss

USE_CUDA = True
all_embeddings = pickle.load(open("./data/vnwords_embedded", "rb"))
word2id = pickle.load(open("./data/word2id", "rb"))
class2id = pickle.load(open("./data/class2id", "rb"))
word2class = pickle.load(open("./data/word2class", "rb"))
class2word = pickle.load(open("./data/class2word", "rb"))
epochs = 1000
batch_size = 20
lr = .15
num_lexemes = sum([len(c) for c in word2class.values()])
alpha = .4
beta = .4

# Chunk into batches of dims
print(all_embeddings.size()[0])
embed_batches = [all_embeddings[:, i:i + batch_size]\
    for i in range(0, all_embeddings.size()[1], batch_size)]

#embed_batch1 = all_embeddings[:, 260:280]
#embed_batch2 = all_embeddings[:, 280:300]
#embed_batches = [embed_batch1, embed_batch2]

for d, embeds in enumerate(embed_batches):
    dim_start = d * batch_size
    dim_end = (d+1) * batch_size
    
    print("Initializing AutoEncoder for dims %i to %i"\
          % (dim_start, dim_end))
    model = AutoEncoder(embeds, word2class, class2word, USE_CUDA)
    model = model.cuda() if USE_CUDA else model
    
    # Ignore any parameters with requires_grad = False
    # aka the pretrained embeddings
    params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)

    num_dims = model.embedding_dims
    num_words = model.num_words
    num_classes = model.num_classes
    last_loss = float("inf")

    for i in range(epochs):
        print("Epoch %i" % i)

        optimizer.zero_grad()

        W_out, C = model.forward()

        # Transpose to match n x W
        W_in = model.W_in
        loss = autoExtend_loss(W_out, W_in, model.E, model.D, C,\
            num_words, num_classes, num_lexemes, num_dims, alpha, beta)
        print("LOSS: %4f" % loss.data[0])
        if loss.data[0] < last_loss:# and i > 5:
            print("Normalizing Maps...")
            model.normalize_columns()

            print("Computing backward...")
            loss.backward()

            optimizer.step()
            last_loss = loss.data[0]
    
    W = model.W_in
    E = model.E.data
    D = model.D.data

    torch.save(W, "./parameters/scoped_embeds.%i_%i"\
               % (dim_start, dim_end))
    torch.save(E, "./parameters/E.%i_%i"\
               % (dim_start, dim_end))
    torch.save(D, "./parameters/D.%i_%i"\
        % (dim_start, dim_end))
