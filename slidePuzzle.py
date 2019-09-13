from heapClassSlide import Heap
import math
import functools
import random
from copy import deepcopy
import csv

class slidePuzzle:
    def __init__(self, inp, skipVerification = False):
        if type(inp) is list:

            if skipVerification is False:
                validState = self.validatePuzzle(inp)
                if validState is False:
                    print('invalid state')

            self._puzzle = inp

            for i,x in enumerate(inp):
                if x == 0:
                    self._blankIndex = int(i)
                    break
            
            self._size = self._sqrt(len(inp))
            self._validMovesArr = self.validMoves(self._blankIndex)
        elif type(inp) is int:
            k = inp ** 2
            self._puzzle = list(range(1,k+1))
            self._puzzle[-1] = 0
            self._blankIndex = k - 1
            self._size = inp
            self._validMovesArr = self.validMoves(self._blankIndex)
        
        self._idealHor = [0] * int(self._size ** 2)
        for i in range(len(self._puzzle)):
            n = self._size * (i % self._size) + (i // self._size)
            self._idealHor[n] = i + 1
    
    def xor(self, a, b):
        return ((a and not b) or (not a and b))

    
    def _sqrt(self, n):
        sqrt = math.sqrt(n)
        q = abs(sqrt - math.floor(sqrt))
        if q >= 0.001:
            return "invalid size"
        sqrt -= q
        return int(sqrt)
    
    def validMoves(self, inp = False):
        q = False
        if inp is False:
            q = True
            inp = self._blankIndex
        
        moves = []
        blankRowPos = inp % self._size
        blankRow = inp // self._size

        if blankRow is not 0:
            moves.append('D')
        
        if blankRow is not (self._size - 1):
            moves.append('U')
        
        if blankRowPos is not 0:
            moves.append('R')
        
        if blankRowPos is not (self._size - 1):
            moves.append('L')
        
        if q is True:
            self._validMovesArr = moves
        else:
            return moves
    
    def validatePuzzle(self, inp):
        sqrt = self._sqrt(len(inp))

        oddSize = (sqrt % 2) == 1

        inversions = self._inversionCount(inp)

        invCountIsOdd = inversions % 2 == 1

        if oddSize is True:
            return not invCountIsOdd
        else:
            for i in range(len(inp)):
                if inp[i] == 0:
                    blankSpot = i
                    break
        
        row = sqrt - blankSpot // sqrt
        rowIsOdd = (row % 2) == 1

        return self.xor(rowIsOdd, invCountIsOdd)
    
    def _inversionCount(self, inp = -1, horizontal = False):
        if inp == -1:
            inp = self._puzzle

        appeared = []
        inversions = 0

        if horizontal is False:
            for i in range(len(inp)):
                j = inp[i]
                if j == 0:
                    continue
                
                r = functools.reduce(lambda a,b: a + 1 if b>j else a, appeared, 0)

                inversions += r
                appeared.append(j)
        else:
            for i in range(len(inp)):
                n = (self._size * (i % self._size)) + (i // self._size)
                j = self._idealHor[inp[n] - 1]
                
                if j == 16:
                    continue
                
                r = functools.reduce(lambda a, b: a+1 if b>j else a, appeared, 0)

                inversions += r
                appeared.append(j)
        
        return inversions
    
    def _swap(self, tlist, a, b):
        tlist[a], tlist[b] = tlist[b], tlist[a]
    
    def slideR(self, inp = False, blankIndex = -1, validMoves = -1):
        if blankIndex == -1:
            blankIndex = self._blankIndex
        
        if validMoves == -1:
            validMoves = self._validMovesArr
        
        q = False

        blankIndexCopy = blankIndex

        if inp is False:
            q = True
            inp = self._puzzle
        
        if 'R' in validMoves:
            self._swap(inp, blankIndex, blankIndex - 1)
        
            if q is True:
                self._blankIndex -= 1
                self.validMoves()
            else:
                blankIndexCopy -= 1
                return [blankIndexCopy, inp]
    
    def slideL(self, inp = False, blankIndex = -1, validMoves = -1):
        if blankIndex == -1:
            blankIndex = self._blankIndex
        
        if validMoves == -1:
            validMoves = self._validMovesArr
        
        q = False

        blankIndexCopy = blankIndex

        if inp is False:
            q = True
            inp = self._puzzle
        
        if 'L' in validMoves:
            self._swap(inp, blankIndex, blankIndex + 1)
        
            if q is True:
                self._blankIndex += 1
                self.validMoves()
            else:
                blankIndexCopy += 1
                return [blankIndexCopy, inp]
    
    def slideU(self, inp = False, blankIndex = -1, validMoves = -1):
        if blankIndex == -1:
            blankIndex = self._blankIndex
        
        if validMoves == -1:
            validMoves = self._validMovesArr
        
        q = False

        blankIndexCopy = blankIndex

        if inp is False:
            q = True
            inp = self._puzzle
        
        if 'U' in validMoves:
            self._swap(inp, blankIndex, blankIndex + self._size)
        
            if q is True:
                self._blankIndex += self._size
                self.validMoves()
            else:
                blankIndexCopy += self._size
                return [blankIndexCopy, inp]
    
    def slideD(self, inp = False, blankIndex = -1, validMoves = -1):
        if blankIndex == -1:
            blankIndex = self._blankIndex
        
        if validMoves == -1:
            validMoves = self._validMovesArr
        
        q = False

        blankIndexCopy = blankIndex

        if inp is False:
            q = True
            inp = self._puzzle
        
        if 'D' in validMoves:
            self._swap(inp, blankIndex, blankIndex - self._size)
        
            if q is True:
                self._blankIndex -= self._size
                self.validMoves()
            else:
                blankIndexCopy -= self._size
                return [blankIndexCopy, inp]
    
    def shuffle(self, moveLimit = 60, variance = 0.3):
        if variance > 0.6:
            variance = 0.6
        elif variance < 0:
            variance = 0
        
        moveVar = math.floor(moveLimit * variance)
        rand = random.randint(0,moveVar * 2 - 1)
        i = moveLimit - moveVar + rand

        while i > 0:
            i -= 1

            r = random.randint(0, len(self._validMovesArr) - 1)
            s = self._validMovesArr[r]

            if s == 'R':
                self.slideR()
            elif s == 'L':
                self.slideL()
            elif s == 'U':
                self.slideU()
            elif s == 'D':
                self.slideD()
    
    def findH(self, inp = -1, blankIndex = False, lastIndex = False, acc = 0, vert = 0, hor = 0):
        if inp == -1:
            inp = self._puzzle
        
        isVert = abs(blankIndex - lastIndex) != 1

        if acc == 0:
            for i,curr in enumerate(inp):
                if curr != 0:
                    intendedRow = (curr - 1) // self._size
                    currentRow = i // self._size
                    intendedCol = (curr - 1) % self._size
                    currentCol = i % self._size

                    rowDist = abs(intendedRow - currentRow)
                    colDist = abs(intendedCol - currentCol)

                    acc += rowDist + colDist

                    li = list(range(self._size))

                    if rowDist == 0:
                        for j in li:
                            testCol = j

                            k = currentRow * self._size + testCol

                            intendTestRow = (inp[k] - 1) // self._size

                            if currentRow != intendTestRow:
                                continue
                            
                            if testCol < currentCol:
                                if inp[k] > curr:
                                    acc += 2
                            else:
                                if inp[k] < curr and inp[k] is not 0:
                                    acc += 2

                    if colDist == 0:
                        for j in li:
                            testRow = j
                            k = currentCol + testRow * self._size
                            intendTestCol = (inp[k] - 1) % self._size

                            if currentCol != intendTestCol:
                                continue

                            if testRow < currentRow:
                                if inp[k] > curr:
                                    acc += 2
                            else:
                                if inp[k] < curr and inp[k] is not 0:
                                    acc +=2
        
        else:
            moved = inp[lastIndex]
            li = list(range(self._size))

            if isVert:
                intendedRow = (moved - 1) // self._size
                currCol = lastIndex % self._size
                currRow = lastIndex // self._size
                lastRow = blankIndex // self._size
                rowDist = abs(intendedRow - currRow)
                prevRowDist = abs(intendedRow - lastRow)

                if rowDist > prevRowDist:
                    acc += 1
                else:
                    acc -= 1
                
                if rowDist == 0:
                    for i in li:
                        testCol = li[i]
                        k = currRow * self._size + testCol

                        intendTestRow = (inp[k] - 1) // self._size

                        if currRow != intendTestRow:
                            continue
                        
                        if testCol < currCol:
                            if inp[k] > moved:
                                acc += 4
                        else:
                            if inp[k] < moved and inp[k] != 0:
                                acc += 4
                elif prevRowDist == 0:
                    for i in li:
                        testCol = i
                        k = lastRow * self._size + testCol

                        intendTestRow = (inp[k] - 1) // self._size
                        if lastRow != intendTestRow:
                            continue
                        
                        if testCol < currCol:
                            if inp[k] > moved:
                                acc -= 4
                        else:
                            if inp[k] < moved and inp[k] != 0:
                                acc -= 4
                            
            else:
                intendedCol = (moved - 1) % self._size
                currCol = lastIndex % self._size
                currRow = lastIndex // self._size
                lastCol = blankIndex % self._size
                colDist = abs(intendedCol - currCol)
                prevColDist = abs(intendedCol - lastCol)

                if colDist > prevColDist:
                    acc += 1
                else:
                    acc -= 1
                
                if colDist == 0:
                    for i in li:
                        testRow = i
                        k = currCol + testRow * self._size
                        intendTestCol = (inp[k] - 1) % self._size

                        if currCol != intendTestCol:
                            continue
                        
                        if testRow < currRow:
                            if inp[k] > moved:
                                acc += 4
                        else:
                            if inp[k] < moved and inp[k] != 0:
                                acc += 4
                elif prevColDist == 0:
                    for i in li:
                        testRow = i
                        k = lastCol + testRow * self._size

                        intendTestCol = (inp[k] - 1) % self._size

                        if lastCol != intendTestCol:
                            continue
                        
                        if testRow < currRow:
                            if inp[k] > moved:
                                acc -= 4
                        else:
                            if inp[k] < moved and inp[k] != 0:
                                acc -= 4
        
        inversions = 0
        horInversions = 0

        if isVert:
            inversions = self._inversionCount(inp.copy())
            vert = inversions // (self._size - 1) + inversions % (self._size - 1)
        
        if not isVert:
            horInversions = self._inversionCount(inp.copy(), True)
            hor = (horInversions // (self._size - 1)) + (horInversions % (self._size - 1))
        
        invertDistance = vert + hor

        return [max([acc, invertDistance]), acc, vert, hor]
    
    def getPuzzle(self):
        return deepcopy(self._puzzle)

    puzzle = property(getPuzzle)

    def getf(self):
        return self._f
    
    f = property(getf)

    def get_blank_index(self):
        return self._blankIndex
    
    blankIndex = property(get_blank_index)

    def get_MD(self):
        return self._MD
    
    MD = property(get_MD)
    
    def solve(self):
        self._h, self._MD, self._invVert, self._invHor = self.findH()
        self._g = 0
        self._f = self._g + self._h
        print ('solving. Initial H value: ', self._h)

        heap = Heap([],'max','g','f')

        maxDepth = 3
        nodes = 0

        if self._size % 2 == 1:
            i = self._blankIndex % 2
        else:
            i = self._blankIndex % 2 + (self._blankIndex // self._size) % 2
        
        while i < maxDepth:
            print('depth: ', i, end='\r')

            res = self._exploreStates(i, heap)
            nodes += res[1]

            if res[0] is False:
                maxDepth +=2
                i += 2
                continue
            else:
                print('nodes explored: ', place_value(nodes), 'max depth: ', i)
                return res[0]
    
    def _exploreStates(self, maxDepth, heap):

        initState = {
            'state': self._puzzle,
            'g': 0,
            'h': self._h,
            'MD': self._MD,
            'invVert': self._invVert,
            'invHor': self._invHor,
            'moves': [],
            'validMoves': self._validMovesArr.copy(),
            'blankIndex': self._blankIndex
        }
        initState['f'] = initState['g'] + initState['h']

        heap.insert(initState)
        
        nodes = 0

        while heap.size > 0:
            currentState = heap.pop()
            if currentState['moves'] == ['R', 'D']:
                # print(currentState)
                found = True

            nodes += 1
            if currentState['f'] > maxDepth:
                continue
            
            currentState['validMoves'] = self.validMoves(currentState['blankIndex'])

            if currentState['h'] == 0:
                return [currentState, nodes]
            
            movelength = len(currentState['moves'])

            if ('R' in currentState['validMoves']) and (movelength == 0 or currentState['moves'][movelength - 1] != 'L'):
                potentialBlank, potentialMove = self.slideR(currentState['state'].copy(),currentState['blankIndex'], currentState['validMoves'].copy())

                hvals = self.findH(potentialMove, potentialBlank, currentState['blankIndex'], currentState['MD'], currentState['invVert'], currentState['invHor'])

                potentialState = {}
                potentialState.clear()

                potentialState = {
                    'state': potentialMove,
                    'g': currentState['g'] + 1,
                    'h': hvals[0],
                    'MD': hvals[1],
                    'invVert': hvals[2],
                    'invHor': hvals[3],
                    'moves': currentState['moves'].copy(),
                    'blankIndex': potentialBlank
                }
                potentialState['moves'].append('R')
                potentialState['f'] = potentialState['g'] + potentialState['h']

                heap.insert(potentialState)

            if ('L' in currentState['validMoves']) and (movelength == 0 or currentState['moves'][movelength - 1] != 'R'):
                potentialBlank, potentialMove = self.slideL(currentState['state'].copy(),currentState['blankIndex'], currentState['validMoves'].copy())

                hvals = self.findH(potentialMove, potentialBlank, currentState['blankIndex'], currentState['MD'], currentState['invVert'], currentState['invHor'])

                potentialState = {}
                potentialState.clear()

                potentialState = {
                    'state': potentialMove,
                    'g': currentState['g'] + 1,
                    'h': hvals[0],
                    'MD': hvals[1],
                    'invVert': hvals[2],
                    'invHor': hvals[3],
                    'moves': currentState['moves'].copy(),
                    'blankIndex': potentialBlank
                }
                potentialState['moves'].append('L')
                potentialState['f'] = potentialState['g'] + potentialState['h']

                heap.insert(potentialState)

            if ('U' in currentState['validMoves']) and (movelength == 0 or currentState['moves'][movelength - 1] != 'D'):
                potentialBlank, potentialMove = self.slideU(currentState['state'].copy(),currentState['blankIndex'], currentState['validMoves'].copy())

                hvals = self.findH(potentialMove, potentialBlank, currentState['blankIndex'], currentState['MD'], currentState['invVert'], currentState['invHor'])

                potentialState = {}
                potentialState.clear()

                potentialState = {
                    'state': potentialMove,
                    'g': currentState['g'] + 1,
                    'h': hvals[0],
                    'MD': hvals[1],
                    'invVert': hvals[2],
                    'invHor': hvals[3],
                    'moves': currentState['moves'].copy(),
                    'blankIndex': potentialBlank
                }
                potentialState['moves'].append('U')
                potentialState['f'] = potentialState['g'] + potentialState['h']

                heap.insert(potentialState)

            if ('D' in currentState['validMoves']) and (movelength == 0 or currentState['moves'][movelength - 1] != 'U'):
                potentialBlank, potentialMove = self.slideD(currentState['state'].copy(),currentState['blankIndex'], currentState['validMoves'].copy())

                # print('potential move: ', potentialMove)
                # print ('potential Blank: ', potentialBlank)
                # print('last blank: ', currentState['blankIndex'])
                # print('MD: ', currentState['MD'])
                # print('invVert: ', currentState['invVert'])
                # print('invHor: ', currentState['invHor'])

                hvals = self.findH(potentialMove, potentialBlank, currentState['blankIndex'], currentState['MD'], currentState['invVert'], currentState['invHor'])

                potentialState = {}
                potentialState.clear()

                potentialState = {
                    'state': potentialMove,
                    'g': currentState['g'] + 1,
                    'h': hvals[0],
                    'MD': hvals[1],
                    'invVert': hvals[2],
                    'invHor': hvals[3],
                    'moves': currentState['moves'].copy(),
                    'blankIndex': potentialBlank
                }
                potentialState['moves'].append('D')
                potentialState['f'] = potentialState['g'] + potentialState['h']

                # print(potentialState)

                heap.insert(potentialState)
            
        return [False, nodes]

    def move_list(self, li):
        boardstate_list = []
        for i in li:
            oldstate = deepcopy(self._puzzle)
            if i is 'U':
                self.slideU()
            elif i is 'D':
                self.slideD()
            elif i is 'R':
                self.slideR()
            elif i is 'L':
                self.slideL()

            boardstate_list.append([*oldstate,i])

        return boardstate_list    


def generate_csv(inp = 4, movelist = []):
    q = slidePuzzle(inp)
    if inp is 4:
        q.shuffle(1000)
        p = q.solve()
        movelist = p['moves']
    
    biglist = q.move_list(movelist)

    try:
        f = open('puzzlelist.csv')
        f.close()
    except:
        with open('puzzlelist.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 'move'])
    
    with open('puzzlelist.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(biglist)

def place_value(n):
    return ("{:,}".format(n))

def convert_solution(li, sol):
    q = slidePuzzle(li)
    sol = sol.split(' | ')
    sol[0] = sol[0][2:]
    del sol[-1]
    sol = list(map(lambda x: int(x), sol))
    
    def map_adj(lis, blank_index):
        blank_col = blank_index % 4
        blank_row = blank_index // 4
        avail_map = {}
        if blank_row is not 0:
            n = str(lis[blank_index - 4])
            avail_map[n] = 'D'
        if blank_row is not 3:
            n = str(lis[blank_index + 4])
            avail_map[n] = 'U'
        if blank_col is not 0:
            n = str(lis[blank_index - 1])
            avail_map[n] = 'R'
        if blank_col is not 3:
            n = str(lis[blank_index + 1])
            avail_map[n] = 'L'
        
        return avail_map
    
    moves = []
    for i in sol:
        r = map_adj(q.puzzle, q.blankIndex)
        k = r[str(i)]
        moves.append(k)
        
        if k is 'U':
            q.slideU()
        elif k is 'D':
            q.slideD()
        elif k is 'R':
            q.slideR()
        elif k is 'L':
            q.slideL()
    
    return moves

        

# q = slidePuzzle([0, 1, 8, 2, 7, 5, 11, 15, 4, 9, 6, 13, 12, 14, 10, 3])
# print(q._idealHor)
# print(q.findH())
# print(q._inversionCount())
# print(q._inversionCount(horizontal= True))
# print(q.solve())
for i in range(300):
    print("""Puzzle number """ + str(i + 1))
    generate_csv()

