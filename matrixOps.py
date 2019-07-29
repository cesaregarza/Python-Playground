from copy import deepcopy
import time
from fractions import Fraction

class Polynomial:
    def __init__(self, array):
        self.poly = []
        for i in array:
            self.poly.append(Fraction(str(i)))
    
    def __add__(self, other):
        if type(other) is not Polynomial:
            other = Polynomial([other])

        if len(self.poly) > len(other.poly):
            large = self.poly
            small = other.poly
        else:
            large = other.poly
            small = self.poly
        
        newPoly = []

        offset = len(large) - len(small)

        for i in range(offset):
            newPoly.append(large[i])
        
        for i in range(len(small)):
            _sum = large[offset + i] + small[i]
            newPoly.append(_sum)
        
        newPoly = Polynomial(newPoly)
        newPoly.reducePower()
        return newPoly
    
    def __sub__(self, other):
        otherPolyArray = []
        for i in other.poly:
            otherPolyArray.append(i * -1)
        
        otherPoly = Polynomial(otherPolyArray)
        return self + otherPoly
    
    def __mul__(self, other):
        if type(other) is not Polynomial:
            other = Polynomial([other])

        selfdegree = len(self.poly) - 1
        otherdegree = len(other.poly) - 1

        if selfdegree > otherdegree:
            large = self.poly
            small = other.poly
        else:
            large = other.poly
            small = self.poly
        
        bigArr = []
        for i in range(len(small)):
            newRow = [0] * (selfdegree + otherdegree + 1)

            for j in range(len(large)):
                index = i + j
                newRow[index] = large[j] * small[i]
            newRowPoly = self.__class__(newRow)
            bigArr.append(newRowPoly)
        
        total = [0] * (selfdegree + otherdegree + 1)
        totalPoly = self.__class__(total)
        for i in bigArr:
            totalPoly += i
        
        totalPoly.reducePower()
        return totalPoly
    
    def __floordiv__(self, other):
        if type(other) is not Polynomial:
            other = Polynomial([other])
        
        selfdegree = len(self.poly) - 1
        otherdegree = len(other.poly) - 1

        if otherdegree > selfdegree:
            return Polynomial([1])
        
        polyArr = [0] * (selfdegree - otherdegree + 1)
        k = self.poly[0] / other.poly[0]
        polyArr[0] = k
        for i in range(selfdegree - 1):
            for j in range(otherdegree):
                n = other.poly[j+1]*k
                k = float(-n + self.poly[i+1])
                polyArr[i+j+1] = k

        polyHolder = Polynomial(polyArr)
        polyHolder.reducePower()
        return polyHolder


    
    def __pow__(self, other):
        p = Polynomial([1] * len(self.poly))
        for i in range(other):
            p *= self
        
        return p

    
    def printPolynomial(self, variable = "x"):
        polyArray = self.poly
        ret = ""
        l = len(polyArray)
        for i in range(l):
            coefficient = polyArray[i]
            if (coefficient == 0):
                continue
            
            if i > 0:
                ret += " + "
            
            ret += str(coefficient)
            if i < (l - 1) :
                ret += " " + str(variable)
            
            if i < (l - 2):
                ret += "^" + str(l - i - 1)

        
        return ret
    
    def horner(self, polyArray, n):
        l = len(polyArray) - 1
        if l == 0:
            return polyArray[0]
        elif l < 0:
            return Fraction(0)
        p = Fraction(polyArray.pop())

        return p + (self.horner(polyArray, n) * n)

    def evalAt(self, x):
        polyCopy = deepcopy(self.poly)
        return self.horner(polyCopy,x)
    
    def derive(self, poly = -1):
        if poly is -1:
            poly = deepcopy(self.poly)

        for i in range(len(poly)):
            poly[i] *= (len(poly) - i - 1)
        
        if len(poly) is 0:
            return Polynomial([0])
        poly.pop()
        return self.__class__(poly)
    
    def nextRootIteration(self, guess):
        zero = self.evalAt(guess)

        onePoly = self.derive()

        if len(onePoly.poly) is 0:
            one = 1
            twoPoly = Polynomial([0,0,0])
        else:
            one = float(onePoly.evalAt(guess))
            twoPoly = onePoly.derive()
        two = float(twoPoly.evalAt(guess))
        threePoly = twoPoly.derive()
        three = float(threePoly.evalAt(guess))
        fourPoly = threePoly.derive()
        four = float(fourPoly.evalAt(guess))

        if one == 0:
            one = 1

        partOne = -1 * zero / one
        partTwo = (zero ** 2) * two / (2 * one ** 3)
        partThree = zero ** 3 * (one * three - 3 * two ** 2)/(one ** 5)
        partFour = (zero ** 4) * (5 * one * two * three - 9 * two ** 3 - one ** 2 * four)/(12 * one ** 7)

        return guess + partOne + partTwo + partThree + partFour
    
    def findRootPT(self, initGuess, countMax = 10, accuracy = 3):
        counter = 0
        overflowCounter = 0
        currAnswer = initGuess
        prevAnswer = 9999999999999
        condition = True
        while condition:
            counter += 1
            prevAnswer, currAnswer = currAnswer, self.nextRootIteration(currAnswer)

            condition = (counter < countMax and round(abs((currAnswer - prevAnswer)),accuracy + 3) > 10 ** -accuracy)

            if abs(currAnswer - prevAnswer) > 10 ** accuracy:
                overflowCounter += 1
                currAnswer = initGuess - 2 ** overflowCounter
        
        return round(currAnswer,accuracy)
    
    def reducePower(self):
        while self.poly[0] == 0:
            self.poly.pop(0)
        
        return


class Mat:
    def __init__(self, mat, accuracy = 6):
        self.valid = self.validate(mat)
        self.mat = mat
        if self.valid is True:
            self.height = len(mat)
            self.length = len(mat[0])
            self.isSquare = self.height == self.length
            self.accuracy = accuracy
        else:
            print("Invalid Matrix!")
    
    def __add__(self, other):
        if other.height is not self.height or other.length is not self.length:
            print("Incompatible Matrices")
            return
        else:
            newMat = []
            for i in range(self.height):
                newRow = []
                for j in range(self.length):
                    newRow.append(self.mat[i][j] + other.mat[i][j])
                newMat.append(newRow)
            return newMat
    
    def __sub__(self, other):
        if other.height is not self.height or other.length is not self.length:
            print("Incompatible Matrices")
            return
        else:
            newMat = []
            for i in range(self.height):
                newRow = []
                for j in range(self.length):
                    newRow.append(self.mat[i][j] - other.mat[i][j])
                newMat.append(newRow)
            return newMat
    
    def __mul__(self, other):
        newMat = []
        for i in range(self.height):
            newRow = []
            for j in range(self.length):
                newRow.append(other * self.mat[i][j])
            newMat.append(newRow)
        return newMat

    def __matmul__(self, other):
        m, n1 = self.height, self.length
        n2, p = other.height, other.length

        if n1 is not n2:
            print("Incompatible Matrices")
            return
        else:
            newMat = []
            for i in range(m):
                newRow = []
                for j in range(p):
                    total = 0
                    for k in range(n1):
                        n = self.mat[i][k] * other.mat[k][j]
                        nreal = round(n.real, self.accuracy)
                        nimag = round(n.imag, self.accuracy)
                        n = complex(nreal, nimag)
                        total += n
                    newRow.append(total)
                newMat.append(newRow)
            newMatObj = self.__class__(newMat)
            return newMatObj
    
    def __eq__(self, other):
        accumulator = True
        if self.height != other.height or self.length != other.length:
            return False
        
        for i in range(self.height):
            for j in range(self.length):
                diff = self.mat[i][j] - other.mat[i][j]
                diffreal = round(diff.real,self.accuracy)
                diffimag = round(diff.imag, self.accuracy)
                accumulator = accumulator and (diffreal == 0) and (diffimag) == 0
        
        return accumulator

    def validate(self, mat):
        valid = True

        valid = isinstance(mat, list) and isinstance(mat[0], list) and valid

        potlen = len(mat[0])
        for i in mat:
            valid = (potlen == len(i)) and valid
        
        return valid

    def printMat(self):
        if self.valid is False:
            print("You can't print an invalid matrix")
            return
        
        for i in self.mat:
            print(i)
        
        return
    
    def swapElem(self, i, j, adjoint = False):

        if adjoint is False:
            self.mat[i][j], self.mat[j][i] = self.mat[j][i], self.mat[i][j]
        else:
            self.mat[i][j], self.mat[j][i] = self.mat[j][i].conjugate(), self.mat[i][j].conjugate()

        return
    
    def transpose(self):
        if self.isSquare is True:
            for i in range(self.height):
                for j in range(self.length):
                    if i <= j:
                        continue
                    
                    self.swapElem(i,j)
            
            return
        else:
            newMat = []
            for j in range(self.length):
                newList = []
                for i in range(self.height):
                    newList.append(self.mat[i][j])
                newMat.append(newList)
            
            self.mat = newMat
            self.height, self.length = self.length, self.height
            return
    
    def adjoint(self):
        if self.isSquare is True:
            for i in range(self.height):
                for j in range(self.length):
                    if i <= j:
                        continue
                    
                    self.swapElem(i,j, True)
            
            return
        else:
            newMat = []
            for j in range(self.length):
                newList = []
                for i in range(self.height):
                    newList.append(self.mat[i][j].conjugate())
                newMat.append(newList)
            
            self.mat = newMat
            self.height, self.length = self.length, self.height
            return
    
    def removeColumn(self, columnInd, matrix = -1):
        if matrix is -1:
            matrix = self.mat
        copy = deepcopy(matrix)
        for i in range(len(matrix)):
            copy[i].pop(columnInd)
        
        return copy
    
    def removeRow(self, rowInd, matrix = -1):
        if matrix is -1:
            matrix = self.mat
        copy = deepcopy(matrix)
        copy.pop(rowInd)
        return copy

    
    def determinant(self, matrix = -1):
        if matrix is -1:
            matrix = self.mat

        if self.isSquare is False:
            print("Determinants can only be applied to square matrices")
            return 0
        
        values = []
        if len(matrix) is 2:
            det = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
            return det

        for i in range(len(matrix[0])):
            matCopy = deepcopy(matrix)

            copy = self.removeColumn(i, matCopy[1:])
            mult = 1
            if i % 2 is 1:
                mult = -mult
            det = self.determinant(copy)
            val = matCopy[0][i] * det * mult
            values.append(val)
        
        sumTot = 0
        for i in values:
            sumTot = i + sumTot
        
        return sumTot
    
    def invert(self):
        det = self.determinant()
        if det is 0:
            print("Matrix cannot be inverted")
            return
        else:
            newMat = []
            for i in range(self.height):
                newRow = []
                for j in range(self.length):
                    subMat = self.removeColumn(j)
                    subMat = self.removeRow(i, subMat)
                    if self.height is 2:
                        det2 = subMat[0][0]
                    else:
                        det2 = self.determinant(subMat)
                    mult = 1
                    if (i+j) % 2 is 1:
                        mult = -mult
                    newRow.append(mult * det2 / det)
                newMat.append(newRow)
            newMatObj = self.__class__(newMat)
            newMatObj.transpose()
            return newMatObj
    
    def properties(self):
        det = self.determinant()
        propList = []
        if det is 0:
            propList.append("Singular")
        else:
            propList.append("Invertible")
        
        copyMat = deepcopy(self.mat)
        copyMat2 = deepcopy(self.mat)
        copy = self.__class__(copyMat)
        copyAdj = self.__class__(copyMat2)
        copyAdj.adjoint()
        if copy == copyAdj:
            propList.append("Hermitian")
        
        m1 = copy @ copyAdj
        m2 = copyAdj @ copy
        
        if m1 == m2:
            propList.append("Unitary")
        return propList
    
    def commute(self, other):
        m1Mat = deepcopy(self.mat)
        m2Mat = deepcopy(other.mat)

        copy = self.__class__(m1Mat)
        copy2 = self.__class__(m2Mat)

        return copy @ copy2 - copy2 @ copy
    
    def eigenvalue(self, initGuess = 0):
        newMat = self.gaussElim()
        charEq = Polynomial([0,1])

        print(newMat)
        for i in range(self.height):
            print(charEq.poly)
            print(newMat[i][i])
            charEq = Polynomial([-1,newMat[i][i]]) * charEq
        
        print (charEq.poly)
        return charEq
    
    def eigenvalues(self, initGuess = 0, countMax = 10, accuracy = 6):
        eigenlist = []
        
        newPoly = self.eigenvalue(0)
        print(newPoly.printPolynomial())
        print(newPoly.poly)
        eigenlist.append(newPoly.findRootPT(initGuess, countMax, accuracy))
        eigenDiv = Polynomial([1,-eigenlist[-1]])

        newPoly = newPoly // eigenDiv
        while len(newPoly.poly) > 1:
            eigenlist.append(newPoly.findRootPT(initGuess, countMax, accuracy))
            eigenDiv = Polynomial([1, -eigenlist[-1]])
            newPoly = newPoly // eigenDiv
        
        return eigenlist
    
    def gaussElim(self, matrix = -1):
        if matrix is -1:
            matrix = self.mat
        mat = deepcopy(matrix)
        print(mat)
        flipCount = 0
        for i in range(len(mat) - 1):
            pivot = mat[i][i]
            pivotCounter = 1
            print("pivot: ",pivot," i: ", i)
            while pivot == 0:
                if (pivotCounter + i) == len(mat):
                    print("Cannot upper-triangulate correctly")
                    return mat
                mat[i], mat[i+pivotCounter] = mat[i+pivotCounter], mat[i]
                print("flipped!")
                flipCount += 1
                pivotCounter += 1
                pivot = mat[i][i]
            
            for j0 in range(len(mat) - i - 1):
                j = j0+1
                turn = mat[i+j][i]
                if turn == 0:
                    break
                print("turn: ", turn)
                for k in range(len(mat) - i):
                    print("j: ",j," k: ",k)
                    mat[i+j][k+i] -= mat[i][k+i] * turn / pivot
                    print(mat)
            
        
        print(mat)
        return [mat,flipCount]
    
    def gaussDet(self):
        mat, flips = self.gaussElim()
        tot = 1
        for i in range(len(mat)):
            tot *= mat[i][i]
        
        tot *= (-1) * (flips % 2)
        return tot
            

        


            
        
        

# spinZ = Mat([[complex(1),complex(0)],[complex(0),complex(-1)]])
# spinY = Mat([[complex(0),complex(0,-1)],[complex(0,1),complex(0)]])
# spinX = Mat([[complex(0),complex(1)],[complex(1),complex(0)]])


# print(spinX.properties())
# print(spinY.properties())
# print(spinZ.properties())
# print("")
# d = Mat(spinX.commute(spinY))
# d.printMat()


# p = Mat([[2,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,2]])
p = Mat([[0,1,3],[1,2,0],[0,3,4]])
# q = Mat([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
# r = Mat(
#     [
#         [23, 42, 17, 55, 45, 23, 12, 45],
#         [12, 28, 25, 38, 77, 19, 18, 45], 
#         [ 5, 34, 76, 23, 32, 15, 14, 63],
#         [13, 54, 48,  3,  7, 37, 49, 51],
#         [11, 49, 74, 32, 14, 11, 46, 25],
#         [45, 82, 23,  4, 60,  7, 29, 12],
#         [13, 25, 26, 82, 62, 16, 34,  4],
#         [ 1, 91, 56, 48,  9, 15, 27, 15]])
# r.transpose()
# k = 8/(3*3.14159265358979324)
# s = Mat([[1,0,0],[0,1,k],[0,k,1]])
# print(s.eigenvalues(0,500,8))
# print(1-k, 1+k)

# p = Mat([[1,2,5,1],[3,-4,3,-2],[4,3,2,-1],[1,-2,-4,-1]])
# r.transpose()
# start2 = time.time()
# print(r.eigenvalues(0,500,8))
# end2 = time.time()
# print("time: ", end2 - start2)

print(p.gaussDet())