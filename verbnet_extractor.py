import verbnet

verbnet_folder = "./data/verbnet3.3/"
"""
Using the vn api, get the vn graph of relaitonships between
each member, and verb class
"""
def get_stats():
    #vn = verbnet.VerbNetParser(directory=verbnet_folder)
    #vnclasses = vn.get_verb_classes()
    #members = vn.get_members()

    #print("%i lemmas" % len(set([m.name for m in members])))
    #print("%i classes" % len(vnclasses))
    #print("%i class - verb combos" % len(members))
    mem2c, c2mem = extract_relations()

    print("%i lemmas" % len(set([m for m in mem2c.keys()])))
    print("%i classes" % len(c2mem.keys()))
    print("%i class - verb combos" % sum([len(v) for v in c2mem.values()]))

def get_highly_distributed_verbs():
    mem2c, c2mem = extract_relations()

    return [m for m, c in mem2c.items() if len(c) > 5]
        

def extract_relations():
    vn = verbnet.VerbNetParser(directory=verbnet_folder)
    members = vn.get_members()
    members2classes = {}
    classes2members = {}

    for m in members:
        members2classes.setdefault(m.name, []).append(m.class_id(subclasses=False))
        classes2members.setdefault(m.class_id(subclasses=False), []).append(m.name)

    return (members2classes, classes2members)

def extract_class_relations():
    """
    We define a class relation as a shared sel_res

    The output is a list of r tuples of (class_1_ID, class_2_ID)
    where r is the number of relations.
    """
    class2selres = {}
    vn = verbnet.VerbNetParser(directory=verbnet_folder)
    for vc in vn.get_verb_classes():
        class2selres[vc.ID] = []
        for themrole in vc.themroles:
            if themrole.soup.SELRESTRS.SELRESTR is not None:
                for sr in themrole.soup.SELRESTRS.find_all("SELRESTR"):
                    if sr.get("Value") == "+":
                        if sr.get("type") not in class2selres[vc.ID]:
                            class2selres[vc.ID].append(sr.get("type"))


    rel_tuples = []
    for class_id1, selres1 in class2selres.items():
        for class_id2, selres2 in class2selres.items():
            if class_id1 != class_id2:
                len(set(selres1).intersection(set(selres2)))
