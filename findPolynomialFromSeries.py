from fractions import Fraction

class Series:
    def __init__(self, array, seriesType = "series"):
        """Create a new Series
        
        Arguments:
            array {list} -- List that contains either the polynomial coefficients in descending power or any sequence of numbers. Accepts numbers or strings in fraction form such as '3/4'
        
        Keyword Arguments:
            seriesType {str} -- Type of series input with the array parameter. Either "solynomial" or "series" (default: {"series"})
        """

        self._poly = []
        self._series = []
        if seriesType == "polynomial":
            for i in array:
                self._poly.append(Fraction(str(i)))
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
            #print("returning!")
            return [Fraction(secondArray[0]), degree]
        
        #print("not returning:")
        #print(secondArray)
        s = self.findDegree(secondArray, degree + 1)
        return s
    
    def generateFactorials(self, n):
        factArray = [1]
        factArray.extend([None] * n)
        for i in range(n):
            factArray[i + 1] = factArray[i] * (i + 1)
        return factArray

    def generatePolySeries(self, polyArray, n):
        series = []
        for i in range(1, n + 1):
            polyArrayCopy = polyArray[:]
            tempHolder = self.horner(polyArrayCopy, i)
            series.append(tempHolder)
        
        return series
    
    def horner(self, polyArray, n):
        l = len(polyArray) - 1
        if l == 0:
            return polyArray[0]
        p = Fraction(polyArray.pop())

        return p + (self.horner(polyArray, n) * n)

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
            
            generatedPolyArray = self.generatePolySeries(polyArray, len(self._series))

            for i in range(len(generatedPolyArray)):
                testArr[i] = self._series[i] - generatedPolyArray[i]

        
        if _i == 20:
            print("Polynomial could not be determined")
        
        self._poly = polyArray
    
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


r = Series([24, 76, 160, 276, 424], "series")
r.findPolynomial()
print(r.printPolynomial())