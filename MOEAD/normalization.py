
def normalize(moead, P):
    length = len(P)
    a = P.max(axis=0)
    b = P.min(axis=0)
    # Weigh parameter
    Max, Min = a[moead.gene_num + 1], b[moead.gene_num + 1]
    if Max > moead.Max:
        moead.Max = Max
    if Min < moead.Min:
        moead.Min = Min
    # std normalize
    for i in range(length):
        P[i][moead.gene_num + 2] = P[i][moead.gene_num]
        P[i][moead.gene_num + 3] = (P[i][moead.gene_num + 1] - moead.Min) / (moead.Max - moead.Min)
    pass


# 记录种群历史的最小值和最小值
def normalize2(moead, P):
    length = len(P)
    a = P.max(axis=0)
    b = P.min(axis=0)
    # std
    Max, Min = a[moead.gene_num+1], b[moead.gene_num + 1]
    moead.Max = Max
    moead.Min = Min
    # std normalize
    for i in range(length):
        P[i][moead.gene_num + 2] = P[i][moead.gene_num]
        P[i][moead.gene_num + 3] = (P[i][moead.gene_num + 1] - moead.Min) / (moead.Max - moead.Min)
    pass
