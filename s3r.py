import spu
import secretflow as sf
import numpy as np
import pdb


sf.shutdown()

sf.init(['sender', 'receiver'], address='local')

cheetah_config = sf.utils.testing.cluster_def(
    parties=['sender', 'receiver'],
    runtime_config={
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM64,
        'enable_pphlo_profile': True,
        'enable_hal_profile':True,
    },
)

def smaller(x, y):
    return (x < y)

def smallerequal(x, y):
    return (x <= y)

def greater(x, y):
    return (x > y)

def greaterequal(x, y):
    return (x >= y)

def equal(x,y):
    return x==y

def notequal(x,y):
    return x!=y

def notquery(x,y):
    return True


def xor(x,y):
    return x^y

spu_device2 = sf.SPU(cheetah_config)
print(spu_device2.cluster_def)

n = 1<<12  
dnum = 7

predicate_operate = [">","<","=","<=",">=","!=","/"]

receiver_operate = ["<",">","=",">=","/","/","<"]

predicate_num = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=dnum, dtype=np.int32)


predicate_matrix = np.tile(predicate_num, (n, 1))


sender_features = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=(n, dnum), dtype=np.int32)


sender, receiver = sf.PYU('sender'), sf.PYU('receiver')


senderdict = {}
senderdict[">"] = []
senderdict[">="] = []
senderdict["<"] = []
senderdict["<="] = []
senderdict["!="] = []
senderdict["="] = []
senderdict["/"] = []

receiverdict = {}
receiverdict[">"] = []
receiverdict[">="] = []
receiverdict["<"] = []
receiverdict["<="] = []
receiverdict["!="] = []
receiverdict["="] = []
receiverdict["/"] = []

def getshare(op,res):
    vector = np.random.randint(0, 2, size=(n, 1), dtype=np.bool_)
    spu_vector = sf.to(receiver,vector)
    sender_res = spu_device2(xor)(res,spu_vector)
    receiverdict[op].append(vector)
    senderdict[op].append(sender_res)


def compare():
    for d in range(0,dnum):
        for op in predicate_operate:
            x = sf.to(sender,sender_features[:,d])
            #pdb.set_trace()
            y = sf.to(receiver,predicate_matrix[:,d])
            if op == ">":
                res = spu_device2(greater)(x,y)
                getshare(op,res)
            elif op == ">=":
                res = spu_device2(greaterequal)(x,y)
                getshare(op,res)
            elif op == "<":
                res = spu_device2(smaller)(x,y)
                getshare(op,res)
            elif op == "<=":
                res = spu_device2(smallerequal)(x,y)
                getshare(op,res)
            elif op == "!=":
                res = spu_device2(notequal)(x,y)
                getshare(op,res)
            elif op == "=":
                res = spu_device2(equal)(x,y)
                getshare(op,res)
            else:
                res = spu_device2(notquery)(x,y)
                getshare(op,res)




compare()




