def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)
    return tbl[i,j]

def main():
    train_seq = []
    i_f = 'Train.dat'
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        if len(line) > 0:
            items = line.split(',')
            train_seq.append(items)
        else:
            f.close()
            break
    test_seq = []
    i_f = 'Test.dat'
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        if len(line) > 0:
            items = line.split(',')
            test_seq.append(items)
        else:
            f.close()
            break
    val_seq = []
    i_f = 'Val.dat'
    f = open(i_f, 'r')
    while 1:
        line = f.readline()
        line = line[0:-1]
        if len(line) > 0:
            items = line.split(',')
            val_seq.append(items)
        else:
            f.close()
            break
    print('Finished reading ....')
    '''
    train_train_edit_dist = []
    for i in range(7000):
        d = []
        for j in range(7000):
            d.append(0)
        train_train_edit_dist.append(d)
    '''
    test_train_edit_dist = []
    for i in range(2000):
        d = []
        for j in range(7000):
            d.append(0)
        test_train_edit_dist.append(d)
    val_train_edit_dist = []
    for i in range(1000):
        d = []
        for j in range(7000):
            d.append(0)
        val_train_edit_dist.append(d)
    '''
    f = open('EDIT_train_kernel.csv', 'w')
    for i in range(len(train_seq)):
        for j in range(i+1, len(train_seq)):
            dist = edit_distance(train_seq[i], train_seq[j])
            train_train_edit_dist[i][j] = dist
            train_train_edit_dist[j][i] = dist

    print('Writing to file ....')
    for i in train_train_edit_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()
    '''
    f = open('EDIT_test_kernel.csv', 'w')
    for i in range(len(test_seq)):
        for j in range(len(train_seq)):
            dist = edit_distance(test_seq[i], train_seq[j])
            test_train_edit_dist[i][j] = dist
    print('Writing to file ....')
    for i in test_train_edit_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()

    f = open('EDIT_val_kernel.csv', 'w')
    for i in range(len(val_seq)):
        for j in range(len(train_seq)):
            dist = edit_distance(val_seq[i], train_seq[j])
            val_train_edit_dist[i][j] = dist

    print('Writing to file ....')
    for i in val_train_edit_dist:
        s = ''
        for j in i:
            s = s + str(j) + ','
        s = s[0:-1]
        f.write(s + '\n')
    f.close()

if __name__ == '__main__':
    main()