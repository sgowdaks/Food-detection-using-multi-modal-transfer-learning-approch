list1 = []
list2 = []

import csv
import random

from pathlib import Path

FILE_PATH = Path('/home/shivani/work/data')

def add_categories(num):
    foods = []
    for i in range(num):
        foods.append(list1[random.randint(0, len(list1)-1)])
    foods.append(list2[random.randint(0, len(list2)-1)])
    return foods


with open(FILE_PATH / 'new_challenge_data.tsv', 'w') as input:
    write = csv.writer(input, delimiter="\t")
    for i in range(800):
        if i % 50 == 0:
            food = add_categories(5)
        elif i % 40 == 0:
            food = add_categories(4)
        elif i % 30 == 0:
            food = add_categories(3)
        elif i % 20 == 0:
            food = add_categories(2)
        else:
            food = add_categories(1)
        write.writerow([food])
        
 
