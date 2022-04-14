import random

total_N = 50
normal_N = 20
abnormal_N = total_N - normal_N

normal_range = 3
abnormal_range = 9

str_list = []
for _ in range(0, normal_N):
    str_list.append('[%d, %d],\n' %
                    (random.randint(0, normal_range-1), random.randint(0, normal_range-1)))

for _ in range(0, abnormal_N):
    str_list.append('[%d, %d],\n' %
                    (random.randint(0, abnormal_range-1), random.randint(0, abnormal_range-1)))


# for _ in range(0, normal_N):
#     str_list.append('[%d],\n' %
#                     (random.randint(0, normal_range-1)))

# for _ in range(0, abnormal_N):
#     str_list.append('[%d],\n' %
#                     (random.randint(0, abnormal_range-1)))


random.shuffle(str_list)

print("".join(str_list))
