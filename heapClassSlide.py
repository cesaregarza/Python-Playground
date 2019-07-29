class Heap:
    def __init__(self, array, order, prop = None, secProp = None):
        self._array = array
        self._order = order
        self._property = prop
        self._secProp = secProp
        self._initHeapify()
    
    def pop(self):
        #Swap first and last
        self._array[0], self._array[-1] = self._array[-1], self._array[0]

        #pop to remove last value which is the highest value
        p = self._array.pop()
        #re-heapify
        self._heapify()
        #return highest value
        return p
    
    def _heapify(self, startIndex = 0):
        maxIndex = len(self._array)
        if startIndex > maxIndex:
            return
        
        child1 = startIndex * 2 + 1
        child2 = child1 + 1
        child1Exists = child1 < maxIndex
        child2Exists = child2 < maxIndex

        finalIndex = -1

        if child1Exists and not self._compare(self._array[startIndex], self._array[child1]):
            if child2Exists and self._compare(self._array[child1], self._array[child2]):
                finalIndex = child1
            elif child2Exists:
                finalIndex = child2
            else:
                finalIndex = child1
        elif child2Exists and not self._compare(self._array[startIndex], self._array[child2]):
            finalIndex = child2
        else:
            return
        
        if finalIndex is not -1:
            self._array[startIndex], self._array[finalIndex] = self._array[finalIndex], self._array[startIndex]
            self._heapify(finalIndex)

    def _compare(self, a, b, equals = False, prop = -1, secCompare = False):
        if prop == -1:
            prop = self._property
        
        if secCompare is True:
            if a[prop] == b[prop]:
                return self._reverseCompare(a, b, equals, self._secProp)
            else:
                return self._compare(a, b, equals)
        
        if prop is None:
            if equals is False:
                if self._order is "max":
                    return a > b
                else:
                    return a < b
            else:
                if self._order is "max":
                    return a >= b
                else:
                    return a <= b
        else:
            if equals is False:
                if self._order is "max":
                    return a[prop] > b[prop]
                else:
                    return a[prop] < b[prop]
            else:
                if self._order is "max":
                    return a[prop] >= b[prop]
                else:
                    return a[prop] <= b[prop]
    
    def _reverseCompare(self, a, b, equals, prop = -1):
        if prop == -1:
            prop = self._property
        
        if prop is None:
            if equals is False:
                if self._order is "max":
                    return a < b
                else:
                    return a > b
            else:
                if self._order is "max":
                    return a <= b
                else:
                    return a >= b
        else:
            if equals is False:
                if self._order is "max":
                    return a[prop] < b[prop]
                else:
                    return a[prop] > b[prop]
            else:
                if self._order is "max":
                    return a[prop] <= b[prop]
                else:
                    return a[prop] >= b[prop]
    
    def _bubbleUp(self, startIndex):
        if startIndex is 0:
            return
        si = startIndex + 1
        parentIndex = int((si - si % 2) / 2 - 1)
        if self._compare(self._array[startIndex], self._array[parentIndex]):
            self._array[startIndex], self._array[parentIndex] = self._array[parentIndex], self._array[startIndex]
        else:
            return
        
        self._bubbleUp(parentIndex)
    
    def insert(self, a):
        self._array.append(a)
        self._bubbleUp(len(self._array) - 1)
    
    def getSize(self):
        return len(self._array)
    
    size = property(getSize)

    def sort(self):
        originalArr = self._array[:]
        sortedArr = []
        while len(self._array):
            sortedArr.append(self.pop())
        
        self._array = originalArr
        return sortedArr
    
    def _initHeapify(self):
        tempArr = self._array[:]
        self._array = []
        while len(tempArr):
            self.insert(tempArr.pop())
        
        return

