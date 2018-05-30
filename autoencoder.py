import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import random
import math

class AutoEncoder(nn.Module):
    def __init__(self, word_embeddings, word2classes, class2words,\
                 USE_CUDA):
        super(AutoEncoder, self).__init__()
        self.embedding_dims = word_embeddings.size(1)
        self.num_embeddings = word_embeddings.size(0)
        self.class2words = class2words
        self.word2classes = word2classes
        # Need to put these tensors on GPU
        self.classes = torch.LongTensor(list(self.class2words.keys()))
        self.classes = self.classes.cuda() if USE_CUDA else self.classes
        self.words = torch.LongTensor(list(self.word2classes.keys()))
        self.words = self.words.cuda() if USE_CUDA else self.words
        self.num_classes = len(self.classes)
        self.num_words = len(self.words)
        if self.num_embeddings != self.num_words:
            raise Exception("you have a graph of %i words, but have passed in %i embeddings..." % (self.num_embeddings, self.num_words))

        # Transpose the embeddings to get n x W, in order
        # to match batching shape
        self.W_in = Variable(word_embeddings.t(), requires_grad=False)
        self.W_in = self.W_in.cuda() if USE_CUDA else self.W_in
        """
        We initialize E as a num_dims x C x W, where
        each dim is independent, and represents the diagonal
        where C[d] and W[d] meet.
        
        We are basically following the dimension-wise equations
        of the paper, where we train batches of dimesnions
        """

        # Hooks for proper gradient updates during backprop
        def _E_hook(grad):
            mask =  Variable(self.E_mask)
            mask = mask.cuda() if USE_CUDA else mask
            return grad * mask

        def _D_hook(grad):
            mask = Variable(self.D_mask)
            mask = mask.cuda() if USE_CUDA else mask
            return grad * mask

        # n x C x W
        # We will set the nonexistent W -C combinations to 0 with a mask
        E_mask = torch.zeros(self.num_classes, self.num_words)
        print("Computing E initializations...")
        # Compute the w - c relaionships where there should indeed
        # be a value
        for c, w_list in class2words.items():
            for w in w_list:
                # Fill the mask where a relationship exists
                E_mask[c, w] = 1

        self.E_mask = E_mask.repeat(self.embedding_dims, 1, 1)
        # We can just initialize D as all ones.
        # Note it will be column normalized
        self.E = nn.Parameter(self.E_mask.clone())
        self.E.register_hook(_E_hook)

        ############################
        ##### DECODER MATRIX D #####
        ############################

        # D is the transpose of E:
        # n x C x W -> n x W x C
        D_mask = E_mask.t()

        self.D_mask = D_mask.repeat(self.embedding_dims, 1, 1)
        # We can just initialize D as all ones.
        # Note it will be normalized across columns
        self.D = nn.Parameter(self.D_mask.clone())
        self.D.register_hook(_D_hook)
        
        # We want to start with normalized columns s.t.
        # word/class pair Eij sums to 1 along the same dim
        self.normalize_columns()
    
    def forward(self):
        """
        Compute the class embeddings of the autoencoder via E
        and use that to compute the W_out via D

        This is DEw per dimension
        """
        print("Computing a Forward pass!")
        """
        We need to only have values where the relation actually exists
        We need to compute a C x n vector here, by taking
        The matrix-vector product of all E[d] CxW matrices
        and W[d] W vectors.
        """
        W = self.W_in
        # n x C x W (mv-product per dim) n x W -> n x C
        C = self.E.bmm(W.unsqueeze(2)).squeeze(2)
        # n x W x C (mv-product per dim n x C -> n x W
        W_out = self.D.bmm(C.unsqueeze(2)).squeeze(2)
        return W_out, C

    def normalize_columns(self):
        """
        Column normalize the Parameters E and D,
        in order to enforce the constraint that 
        the weights E that Map words to classes should sum to 1
        and vice versa, the weights of D that map classes to words
        should sum to 1
        """

        def _column_sums(T):
            T_out = T.data.clone()
            # For all words i in T, the sum of all vectors
            # Tij at dimension d should be 1
            # Note T is a n x C x W batch-matrix
            for i in range(T_out.size()[0]):
                # T[i] is a C x W matrix, colsums is of size W
                # add .001 to avoid divide-by-zero
                colsums = torch.sum(T_out[i], dim=0) + .001
                # Make the W sums vector a W x S matrix
                # by repeating it S times
                colsumsmat = colsums.repeat(T_out[i].size()[0], 1)
                # Perform pointwise division to normalize so
                # each col sums to 1
                T_out[i] = colsumsmat

            return T_out

        self.E.data /= _column_sums(self.E)
        self.D.data /= _column_sums(self.D)
