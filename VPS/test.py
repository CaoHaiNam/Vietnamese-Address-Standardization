import sys
sys.path.append('../Vietnamese-Address-Standardization')
import gc
import Siameser

# std = Siameser.Siameser('AD')

with open('Data/VPS/test_2.txt') as f:
    raw_add = []
    check = 'Before correct:  '
    # check = 'Address before: '
    for line in f:
        line = line.strip('\n')
        if check in line:
            raw_add.append(line[len(check):])
print(raw_add)
outF = open('Data/VPS/result_test_2.txt', "w")
    
std = Siameser.Siameser('AD')
for add in raw_add:
    std_add = std.standardize(add)
    line = ', '.join(list(std_add.values()))
    outF.write('before address: {}\n'.format(add))
    outF.write('after address: {}\n'.format(line))
    gc.collect()
outF.close()
