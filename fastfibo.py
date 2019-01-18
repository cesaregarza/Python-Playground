import time

class Fibo:
    def __init__(self, n):
        self.identity = [[1, 0],[0, 1]]
        self._start = [[1, 1],[1,0]]
        self._memo = [self._start[:]]
        self._n = n - 1
    
    def powersOfTwo(self, n):
        array = []
        t = list(bin(n))[2:]
        for i, x in enumerate(t):
            if x is "1":
                array.append(2 ** (len(t) - i - 1))
            else:
                array.append(0)
        
        return array
    
    def _addToMemo(self, ind):
        if (ind - 1) >= len(self._memo):
            self._addToMemo(ind - 1)
        
        #print("index: " + str(ind))
        #print("memo lenght: " + str(len(self._memo)))
        self._memo.append(self.matrixMult(self._memo[ind - 1], self._memo[ind - 1]))
        #print("added memo index: " + str(2 ** ind))
    
    def matrixExp(self, pow):
        pows = self.powersOfTwo(pow)
        startInd = len(pows) - 1
        total = self.identity[:]

        for i, x in enumerate(pows):
            if (startInd - i) > len(self._memo):
                self._addToMemo(startInd - i)
            
            if x is not 0:
                total = self.matrixMult(total, self._memo[startInd - i])
        
        return total
    
    def getVal(self):
        if self._n is -1:
            return 0
        elif self._n < 2:
            return 1
        elif self._n is 2:
            return 2
        elif self._n is 3:
            return 3
        else:
            return self.matrixExp(self._n)[0][0]
    
    value = property(getVal)

    def matrixMult(self, mat1, mat2):
        m, n = len(mat1), len(mat1[0])
        n1, p = len(mat2), len(mat2[0])

        if n != n1:
            return 'incompatible matrices'
        
        array = []

        for i in range(m):
            tempArr = []
            for j in range(p):
                total = 0
                for k in range(n):
                    total += mat1[i][k] * mat2[k][j]
                
                tempArr.append(total)
            
            array.append(tempArr)
        
        return array


start = time.time()
r = Fibo(4000)
print(r.value)
end = time.time()
print("time: " + str(end - start))