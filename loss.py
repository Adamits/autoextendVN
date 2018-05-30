import torch
from torch.autograd import Variable
import numpy as np

def rel_constraint(C, R, r_expected):
    """
    Reduce the sum squared error of the relation matrix R mv_prod C

    So, in practice r_expected it just a vector of zeros
    """
    # n x C x C * n x C --> n x C
    r_out = R.bmm(C.unsqueeze(2)).squeeze(2)
    diffs = r_out - r_expected
    cost = torch.sum(diffs**2)

    return cost

def word_constraint(w_out, w_in):
    """
    w_out is a n x W matrix, as is w_in

    return the sum squared error of w_out and w_in
    """

    diffs = w_out - w_in
    cost = torch.sum(diffs**2)

    return cost

def lexeme_constraint(W, E, D, C):
    """
    Align lexemes in E with lexemes in D.

    Return sum squared error between encoder
     lexemes and decoder lexemes
    """
    # n x C x W * n x W -> n x C x W
    l1 = E * W.unsqueeze(1)
    # n x (W x C)^T
    l2 = (D * C.unsqueeze(1)).permute(0, 2, 1)

    # By default this should flatten the tensor
    # and then take the l2 norm
    diffs = l1 - l2
    cost = torch.sum(diffs**2)

    return cost

def autoExtend_loss(w_out, w_pretrained, E, D, C,\
            num_words, num_classes, num_lexemes, num_dims, alpha, beta):
    """
    w_out: a pytorch Variable of size w x embedding dims.
        This is the output of the autoEncoder
    w_pretrained: a pytorch Variable of size w x embedding dims, with 
        pretrained values. This is the input to the autoEncoder
    C: The 'synset' (inour case VN class) vector computed by E * W
    E: The mapping tensor E from W_in to C
    D: The mapping tensor D from C to W_out
    """
    print(E.data.nonzero().size()[0])

    w_constraint = word_constraint(w_out, w_pretrained)\
                            / num_words
    l_constraint = lexeme_constraint(w_pretrained, E, D, C)\
                            / num_lexemes
    print("word constraint, lexeme constraint")
    print(torch.sum(w_constraint.data), torch.sum(l_constraint.data))
    # Apply hyperparameters and sum, note relation constraint is unused
    return torch.sum((w_constraint * alpha) + (l_constraint * beta))#\
       # + (r_constraint * (1-alpha+beta))

