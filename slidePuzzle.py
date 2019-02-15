from heapClassSlide import Heap

class slidePuzzle:
    def __init__(self, inp, skipVerification = False):
        if type(inp) == "list":
            self._puzzle = inp

            for i,x in enumerate(inp):
                if x == 0:
                    self._blankIndex = int(i)
                    break
            
            self._size = len(inp)