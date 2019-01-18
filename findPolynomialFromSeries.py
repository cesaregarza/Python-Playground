from fractions import Fraction
import matplotlib.pyplot as plt
import time

class Polynomial:
    def __init__(self, array):
        """Create a new Polynomial
        
        Arguments:
            array {list} -- Coefficients of the polynomial
        """
        self._poly = []
        for i in array:
            self._poly.append(Fraction(str(i)))
        
    def __add__(self, other):
        if len(self._poly) > len(other._poly):
            large = self._poly
            small = other._poly
        else:
            large = other._poly
            small = self._poly
        
        newPoly = []

        offset = len(large) - len(small)
        
        for i in range(offset):
            newPoly.append(large[i])
        
        for i in range(len(small)):
            _sum = large[offset + i] + small[i]
            newPoly.append(_sum)
        
        return Polynomial(newPoly)
    
    def __sub__(self, other):
        otherPolyArray = []
        for i in other._poly:
            otherPolyArray.append(i * -1)
        
        otherPoly = Polynomial(otherPolyArray)
        return self + otherPoly

    def evaluateSeries(self, n):
        series = []

        for i in range(1, n+1):
            polyCopy = self._poly[:]
            series.append(self.horner(polyCopy, i))
        
        return series
    
    def evalOne(self, t):
        polyCopy = self._poly[:]
        return self.horner(polyCopy, t)
    
    def evalMult(self, eList):
        series = []
        for i in eList:
            series.append(self.evalOne(i))
        
        return series
    
    def horner(self, polyArray, n):
        l = len(polyArray) - 1
        if l == 0:
            return polyArray[0]
        p = Fraction(polyArray.pop())

        return p + (self.horner(polyArray, n) * n)

    def printPolynomial(self, variable = "x"):
        polyArray = self._poly
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
    
    def plot(self, resolution = 25):
        plt.figure(1)
        plt.subplot(111)
        t = []
        for i in range(resolution):
            t.append(i)

        plt.plot(t, self.evalMult(t), 'k', label = self.printPolynomial())
        plt.legend()
        plt.show()


class Series:
    def __init__(self, array, seriesType = "series"):
        """Create a new Series
        
        Arguments:
            array {list} -- List that contains either the polynomial coefficients in descending power or any sequence of numbers. Accepts numbers or strings in fraction form such as '3/4'
        
        Keyword Arguments:
            seriesType {str} -- Type of series input with the array parameter. Either "polynomial" or "series" (default: {"series"})
        """

        self._polyArr = []
        self._series = []
        if seriesType == "polynomial":
            for i in array:
                self._polyArr.append(Fraction(str(i)))
            self._poly = Polynomial(self._polyArr)
        elif seriesType == "series":
            for i in array:
                self._series.append(Fraction(str(i)))
        else:
            print("invalid type")
    
    def findDegree(self, array, degree = 1):

        secondArray = []
        if len(array) == 1:
            return False
        
        accumulator = True
        for i in range(1, len(array)):
            g = array[i] - array[i - 1]
            secondArray.append(g)
            if i == 1:
                continue
            accumulator = accumulator and (secondArray[i - 1] == secondArray[i - 2])
        
        if accumulator:
            return [Fraction(secondArray[0]), degree]
        
        s = self.findDegree(secondArray, degree + 1)
        return s
    
    def generateFactorials(self, n):
        factArray = [1]
        factArray.extend([None] * n)
        for i in range(n):
            factArray[i + 1] = factArray[i] * (i + 1)
        return factArray

    def findPolynomial(self):
        polyArray = []
        factorials = self.generateFactorials(10)
        testArr = self._series[:]
        l = 0
        degree = 999
        _i = 0

        while degree > 0:
            _i += 1
            if _i == 20:
                break
            
            temp = self.findDegree(testArr)
            coefficient = temp[0]
            tempDeg = temp[1]


            if degree == tempDeg:
                tempDeg -= 1
                coefficient = Fraction(testArr[0])
            
            degree = tempDeg
            coefficient /= factorials[degree]
            
            if len(polyArray) == 0:
                polyArray = [Fraction(0)] * (degree + 1)
                l = len(polyArray)
                polyArray[0] = coefficient
            else:
                polyArray[l - degree - 1] = coefficient

            polynomial = Polynomial(polyArray)

            generatedPolyArray = polynomial.evaluateSeries(len(self._series))

            for i in range(len(generatedPolyArray)):
                testArr[i] = self._series[i] - generatedPolyArray[i]

        
        if _i == 20:
            print("Polynomial could not be determined")
        
        self._poly = Polynomial(polyArray)
    
    def printPolynomial(self, variable = "x"):
        return self._poly.printPolynomial(variable)
    
    def plot(self, resolution = 25):
        return self._poly.plot(resolution)


start = time.time()
r = Series([2, 8, 24, 64, 160])
r.findPolynomial()
print(r.printPolynomial())
r.plot()
end = time.time()
print(end - start)