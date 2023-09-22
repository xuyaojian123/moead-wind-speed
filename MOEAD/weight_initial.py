import numpy as np


def load_weight(moead):
    path = 'MOEAD/weights.csv'
    # if os.path.exists(path) == False:
    # if True:
    #     # 生成均值向量
    #     mv = Mean_vector(moead.individual_num, 2, path)
    #     mv.generate()
    #     print('weight factor created')
    W = np.loadtxt(fname=path)
    # moead.Pop_size = W.shape[0]
    moead.W = W
    return W


# Calculate T neighbors for each weighted Wi
def cpt_W_Bi_T(moead):
    for bi in range(moead.W.shape[0]):
        Bi = moead.W[bi]
        DIS = np.sum((moead.W - Bi) ** 2, axis=1)
        B_T = np.argsort(DIS)
        # 第0个是自己（距离永远最小）
        B_T = B_T[1:moead.T_size + 1]
        moead.W_Bi_T.append(B_T)


'''
求解均值向量
'''


class Mean_vector:
    # 对m维空间，目标方向个数H
    def __init__(self, H=5, m=3, path='out.csv'):
        self.H = H
        self.m = m
        self.path = path
        self.stepsize = 1 / H

    def perm(self, sequence):
        # ！！！ 序列全排列，且无重复
        l = sequence
        if (len(l) <= 1):
            return [l]
        r = []
        for i in range(len(l)):
            if i != 0 and sequence[i - 1] == sequence[i]:
                continue
            else:
                s = l[:i] + l[i + 1:]
                p = self.perm(s)
                for x in p:
                    r.append(l[i:i + 1] + x)
        return r

    def get_mean_vectors(self):
        H = self.H
        m = self.m
        sequence = []
        for ii in range(H):
            sequence.append(0)
        for jj in range(m - 1):
            sequence.append(1)
        ws = []

        pe_seq = self.perm(sequence)
        for sq in pe_seq:
            s = -1
            weight = []
            for i in range(len(sq)):
                if sq[i] == 1:
                    w = i - s
                    w = (w - 1) / H
                    s = i
                    weight.append(w)
            nw = H + m - 1 - s
            nw = (nw - 1) / H
            weight.append(nw)
            if weight not in ws:
                ws.append(weight)
        return ws

    def save_mv_to_file(self, mv):
        f = np.array(mv, dtype=np.float64)
        np.savetxt(fname=self.path, X=f)

    def generate(self):
        m_v = self.get_mean_vectors()
        self.save_mv_to_file(m_v)

# mv = Mean_vector(20, 2, 'test.csv')
# mv.generate()
