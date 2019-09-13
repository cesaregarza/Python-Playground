import csv, random

#Hashmap to remap correct move to numbers
hashmap = {
    'U': 0,
    'R': 1,
    'D': 2,
    'L': 3
}
first_row = True
with open('puzzlelist.csv') as input_data:
    with open('traindata_80.csv','w', newline='') as train_data:
        with open('testdata_20.csv', 'w', newline='') as test_data:
            train_writer = csv.writer(train_data)
            test_writer = csv.writer(test_data)

            for row in csv.reader(input_data):
                #Check if row is empty
                if row:
                    if first_row:
                        first_row = False
                        train_writer.writerow(row)
                        test_writer.writerow(row)
                        continue
                    else:
                        for i,x in enumerate(row):
                            #try casting to integer
                            try:
                                x = int(x)
                                row[i] = x/15
                            #if it fails, grab from hashmap
                            except:
                                row[i] = hashmap[x]

                    r = random.random()
                    #ensure split between training and testing data
                    if r <= 0.8:
                        train_writer.writerow(row)
                    else:
                        test_writer.writerow(row)

            print('done!')