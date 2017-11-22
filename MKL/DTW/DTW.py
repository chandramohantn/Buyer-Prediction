from numpy import array, zeros, argmin, inf

def dtw(x, y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """

    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = abs(x[i] - y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

def main():
    i_f = 'Train.dat'
    train_seq = []
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        d = []
        if len(line) > 0:
            items = line.split(',')
            for i in items[1:]:
                d.append(int(i))
            train_seq.append(d)
        else:
            f.close()
            break
    print('Finished reading ....')
    '''
    train_train_dtw_dist = []
    for i in range(7000):
        d = []
        for j in range(7000):
            d.append(0)
        train_train_dtw_dist.append(d)
    f = open('DTW_train_kernel.csv', 'w')
    for i in range(len(train_seq)):
        for j in range(i+1, len(train_seq)):
            dist, cost, acc, path = dtw(seq[i], seq[j])
            train_train_dtw_dist[i][j] = dist
            train_train_dtw_dist[j][i] = dist
    print('Writing to file ....')
    for i in train_train_dtw_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()
    '''
    i_f = 'Test.dat'
    test_seq = []
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        d = []
        if len(line) > 0:
            items = line.split(',')
            for i in items[1:]:
                d.append(int(i))
            test_seq.append(d)
        else:
            f.close()
            break
    print('Finished reading ....')
    test_train_dtw_dist = []
    for i in range(2000):
        d = []
        for j in range(7000):
            d.append(0)
        test_train_dtw_dist.append(d)
    f = open('DTW_test_kernel.csv', 'w')
    for i in range(len(test_seq)):
        for j in range(len(train_seq)):
            dist, cost, acc, path = dtw(test_seq[i], train_seq[j])
            test_train_dtw_dist[i][j] = dist
    print('Writing to file ....')
    for i in test_train_dtw_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()

    i_f = 'Val.dat'
    val_seq = []
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        d = []
        if len(line) > 0:
            items = line.split(',')
            for i in items[1:]:
                d.append(int(i))
            val_seq.append(d)
        else:
            f.close()
            break
    print('Finished reading ....')
    val_train_dtw_dist = []
    for i in range(1000):
        d = []
        for j in range(7000):
            d.append(0)
        val_train_dtw_dist.append(d)
    f = open('DTW_val_kernel.csv', 'w')
    for i in range(len(val_seq)):
        for j in range(len(train_seq)):
            dist, cost, acc, path = dtw(val_seq[i], train_seq[j])
            val_train_dtw_dist[i][j] = dist
    print('Writing to file ....')
    for i in val_train_dtw_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()

if __name__ == '__main__':
    main()
    
