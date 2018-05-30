import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create Embeddings")
    parser.add_argument('batch_size', help='the batch size')

    args = parser.parse_args()
    batch_size = int(args.batch_size)

    Ws = []
    Es = []
    Ds = []

    for i in range(int(300 / batch_size)):
        Es.append(torch.load("./parameters/E.%i_%i"\
                             % (i*batch_size, (i+1) * batch_size)))
        Ds.append(torch.load("./parameters/D.%i_%i"\
                             % (i*batch_size, (i+1) * batch_size)))
        Ws.append(torch.load("./parameters/scoped_embeds.%i_%i"\
                             % (i*batch_size, (i+1) * batch_size)))

    W = torch.FloatTensor(300, Ws[0].size()[1]).cuda()
    E = torch.FloatTensor(300, Es[0].size()[1], Es[0].size()[2]).cuda()
    D = torch.FloatTensor(300, Ds[0].size()[1], Ds[0].size()[2]).cuda()
    
    for i in range(int(300 / batch_size)):
        W[i*batch_size:(i+1)*batch_size, :] = Ws[i].data
        E[i*batch_size:(i+1)*batch_size, :, :] = Es[i]
        D[i*batch_size:(i+1)*batch_size, :, :] = Ds[i]

    C = E.bmm(W.unsqueeze(2)).squeeze(2)
    
    # n x W -> W x n
    W = W.t()
    # n x C x W -> C x W x n
    E = E.permute(1, 2, 0)
    # n x W x C -> W x C x n
    D = D.permute(1, 2, 0)
    #n x W -> W x n
    C = C.t()
    

    torch.save(W, "./embed_mappings/scoped_embeds")
    torch.save(E, "./embed_mappings/E")
    torch.save(D, "./embed_mappings/D")
    torch.save(C, "./embed_mappings/C")
