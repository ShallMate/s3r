import spu
import secretflow as sf
import numpy as np
import pdb
import tenseal as ts
from joblib import Parallel, delayed
import random

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

spu_device2 = sf.SPU(cheetah_config)
sender, receiver = sf.PYU('sender'), sf.PYU('receiver')

n = 1<<20  
dnum = 5

ops = {">":[0, 2, -1],
       ">=":[1,0.5,-0.5],
       "<":[0,-0.5,0.5],
       "<=":[1,-2,1],
       "=":[1,-1.5,0.5],
       } 

symbols = [">", "<", "=", ">=", "<=","/"]

conops = ["AND","OR"]

# dnum 的长度


# 生成 dnum 长的随机符号列表
op_list = [random.choice(symbols) for _ in range(dnum)]

con_list = [random.choice(conops) for _ in range(dnum-1)]

predicate_num = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=dnum, dtype=np.int32)


predicate_matrix = np.tile(predicate_num, (n, 1))


sender_features = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=(n, dnum), dtype=np.int32)

def greater(x, y):
    return (x>y)

def smaller(x, y):
    return (x<y)

def compare(x,y):
    return x.astype(int)+(y.astype(int)<<1)

def sub(x,y):
    return x-y

x = sf.to(sender,sender_features)
y = sf.to(receiver,predicate_matrix)


op_greater = spu_device2(greater)(x,y)
op_smaller = spu_device2(smaller)(x,y)
res = spu_device2(compare)(op_greater,op_smaller)

receivershare = np.random.randint(0,65536, size=(n, dnum), dtype=np.int32)
spu_receivershare = sf.to(receiver,receivershare)

spu_sendershare = spu_device2(sub)(res,spu_receivershare)

sendershare = sf.reveal(spu_sendershare)

poly_modulus_degree = 1 << 15  # 32768，增加多项式模数度
coeff_mod_bit_sizes = [60, 50, 50, 50, 50, 50, 50, 50, 60]  # 保持较大的比特位
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
context.global_scale = 2**64  # 设置全局尺度
context.generate_galois_keys()


# 按列加密 sendershare 和 receivershare
# 通过 transpose 将矩阵按列进行加密
# 设置分块大小
block_size = poly_modulus_degree  

ones_plain = [1] * block_size

# 将矩阵按列分块加密
def block_encrypt(matrix, block_size):
    all_blocks = []
    for col in matrix.T:
        encrypted_blocks = []
        for i in range(0, len(col), block_size):
            block = col[i:i + block_size]
            encrypted_blocks.append(ts.ckks_vector(context, block.tolist()))
        all_blocks.append(encrypted_blocks)
    return all_blocks

# 对 sendershare 和 receivershare 进行分块加密
enc_sendershare_blocks = block_encrypt(sendershare, block_size)
enc_receivershare_blocks = block_encrypt(receivershare, block_size)

# 同态运算加法
enc_x_list = [enc_s + enc_r for enc_s, enc_r in zip(enc_sendershare_blocks, enc_receivershare_blocks)]


# 结果列表
enc_result_list = []

# 假设我们只处理前两列作为示例
for i, enc_x in enumerate(enc_x_list):  # 这里只做示例处理前两列
    enc_list = []
    for enc_xx in enc_x:
        if op_list[i] != "/":
            enc_result = enc_xx.polyval(ops[op_list[i]]) #">"
            enc_list.append(enc_result)
        else:
            enc_result = ts.ckks_vector(context, ones_plain)
            enc_list.append(enc_result)
    enc_result_list.append(enc_list)


sr_res = []


for j in range(len(enc_result_list[0])):
    print(j)
    sr_op_res = enc_result_list[0][j]
    for i in range(1, dnum):
        if con_list[i-1] == "AND":
            # 如果是 AND，则进行乘法操作
            sr_op_res = sr_op_res * enc_result_list[i][j]
        else:
            # 在 ELSE 分支中，保持逻辑不变，但减少乘法操作
            temp_res = sr_op_res * enc_result_list[i][j]
            sr_op_res = (sr_op_res + enc_result_list[i][j]) + temp_res  # 优先处理加法，再做乘法
    sr_res.append(sr_op_res)


               






# 假设 sr_res 是 CKKS 加密向量的列表
decrypted_res = [res.decrypt() for res in sr_res]

# 输出解密后的结果
for idx, result in enumerate(decrypted_res):
    print(f"Decrypted result {idx+1}: {result}")
