import random

normal_N = 20
abnormal_N = 50
N = 50
R = 5
elem_num = 1

str_list = []
for _ in range(0, N):
    for _ in range(0, elem_num):
        str_list.append('[%d, %d],\n' %
                        (random.randint(0, R), random.randint(0, R)))


print("".join(str_list))
