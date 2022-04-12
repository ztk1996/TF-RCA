import random

N = 50
R = 5
elem_num = 2

str_list = []
for _ in range(0, N):
    for _ in range(0, elem_num):
        str_list.append('[%d, %d],\n' %
                        (random.randint(0, R), random.randint(0, R)))


print("".join(str_list))
