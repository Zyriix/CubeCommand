import os
import json
tagSeqPath = "d:/mcube/python/lstm/data/tagSeq"

def loadTagSeq(label):
    rs = []
    for l in label:
        f = open(os.path.join(tagSeqPath, l+'.json'))
        seq = json.load(f)
        rs+=seq
    return rs
