import random
import numpy as np
import matplotlib.pyplot as plt

def run():
    p, pity, pitymax = 0.0253, 0, 30
    hashmap = {}
    count = 0
    boxes = 0
    maxcount = 158
    listmap = list(range(maxcount))
    dupes = False

    def legendGet():
        nonlocal pity, hashmap, maxcount, listmap, dupes, count
        if count >= maxcount:
            return

        if dupes is True:
            test = random.randint(1,maxcount)
            if test not in hashmap:
                hashmap[test] = 1
                count += 1
                return
            else:
                hashmap[test] += 1
                return
        else:
            test = listmap[random.randint(0,len(listmap)) - 1]
            count += 1
            listmap.remove(test)
            return

    while count < maxcount:
        boxes += 1
        obj1 = random.random()
        obj2 = random.random()
        obj3 = random.random()

        acc = False
        for i in [obj1, obj2, obj3]:
            if i < p:
                acc = True
                legendGet()
        
        if acc is True:
            continue
        elif pity is pitymax:
            legendGet()
    
    return boxes

runs = 100000
total = 0
runArray = np.empty(runs)

for i in range(runs):
    one = run()
    runArray[i] = one
    print (i, end="\r")

mean, median, amax, amin = np.mean(runArray), np.median(runArray), np.amax(runArray), np.amin(runArray)

print ("mean:" + str(mean) + " | median: " + str(median))
print("Longest run: " + str(amax) + ", Shortest: " + str(amin))
print("25th Percentile: " + str(np.percentile(runArray, 25)) + " 75th Percentile: " + str(np.percentile(runArray, 75)))

hist, bins = np.histogram(runArray, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.xlabel('Boxes to Complete')
plt.ylabel('Simulations')
plt.title('Apex Legends Unbox all Legendaries')
plt.bar(center, hist, align='center', width=width)
plt.show()