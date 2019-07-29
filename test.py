def run():
    hashmap = {}
    keys = []
    nmax = 6

    for x in range(nmax):
        for y in range(nmax):
            for z in range(nmax):
                En = (x) ** 2 + (y) ** 2 + (z) ** 2
                if En not in hashmap:
                    if x is not 0 and y is not 0 and z is not 0:
                        hashmap[En] = 1
                        keys.append(En)
                        print(x, y, z, En)
                else:
                    if (x == y) and (y == z):
                        continue
                    elif x is not 0 and y is not 0 and z is not 0:
                        hashmap[En] += 1
                        print(x, y, z, En)
    
    keys.sort()
    count = 1
    for i in keys:
        print(count, i, hashmap[i])
        count += 1

run()