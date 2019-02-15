val = 9
acc = True
k = 100000
for i in range(k):
    val = 9 * val
    print ('currently testing 9^' + str(i), end="\r")
    if val % 8 is not 1:
        print("Not 1 found at power " + str(i))
        acc = False

print('')
print('done')