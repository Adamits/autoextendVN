import torch
import pickle

def get_lexeme(w, c, W, E, D):
    word = W[w]
    Eij = E[c, w]
    Dji = D[w, c]
    print(Eij)
    print(Dji)
    l1 = Eij * word
    l2 = Dji * word

   # print("E paramaters: ")
   # print(Eij)
   # print("D Parameters:")
   # print(Dji)
    
    return l2#(l1 + l2) / 2

def cosine_sim(w1, w2):
    return w1.dot(w2) / (torch.norm(w1, 2) * torch.norm(w2, 2))

class2id = pickle.load(open("./class2id", "rb"))
word2id = pickle.load(open("./word2id", "rb"))

W = torch.load("./scoped_embeds")
E = torch.load("./E")
D = torch.load("./D")
C = torch.load("./C")

hack_id = word2id["hack"]
develop_id = word2id["develop"]
get_id = word2id["get"]
give_id = word2id["give"]
cut_class_id = class2id["cut-21.1"]
build_class_id = class2id["build-26.1"]
obtain_class_id = class2id["obtain-13.5.2"]
get_class_id = class2id["get-13.5.1"]

obtain_class = C[obtain_class_id]
get_class = C[get_class_id]

develop = W[develop_id]
assemble = W[word2id["assemble"]]
hack = W[hack_id]
get = W[get_id]
give = W[give_id]

hack_cut = get_lexeme(hack_id, cut_class_id, W, E, D)
hack_build = get_lexeme(hack_id, build_class_id, W, E, D)
develop_build = get_lexeme(develop_id, build_class_id, W, E, D)

print("Cosine sim of hack and develop")
print(cosine_sim(hack, develop))
print("Cosine sim of hack in cut sense, and develop")
print(cosine_sim(hack_cut, develop))
print("Cosine sim of hack in build sense, and develop")
print(cosine_sim(hack_build, develop))
print("Cosine sim of hack in build sense, and develop in the build sense")
print(cosine_sim(hack_build, develop_build))

print("\n Similarities to Class: \n")
print("Cosine sim of give and obtain class")
print(cosine_sim(give, obtain_class))
print("Cosine sim of get and obtain class")
print(cosine_sim(get, obtain_class))
print("Cosine sim of get and get class")
print(cosine_sim(get, get_class))
print("Cosine sim of give and get class")
print(cosine_sim(give, get_class))

"""
run_class_id = class2id["run-51.3.2"]
contig_loc_class_id = class2id["contiguous_location-47.8"]
bound_id = word2id["bound"]
limp_id = word2id["limp"]

run_class = C[run_class_id]
limp = W[limp_id]
bound = W[bound_id]
bound_run = get_lexeme(bound_id, run_class_id, W, E, D)
bound_contig_loc = get_lexeme(bound_id, contig_loc_class_id, W, E, D)

print(cosine_sim(bound_run, bound_contig_loc))

print(cosine_sim(limp, bound_run))
print(cosine_sim(limp, bound_contig_loc))
print(cosine_sim(limp, bound))

print(cosine_sim(run_class, bound_run))
print(cosine_sim(run_class, bound_contig_loc))
print(cosine_sim(run_class, bound))
"""
