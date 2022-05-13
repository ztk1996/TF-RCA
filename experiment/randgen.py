import random
from exp_structure_change import service_changes

total_N = 50
normal_N = 20
abnormal_N = total_N - normal_N

normal_range = 5
abnormal_range = len(service_changes)

str_list = []
for _ in range(normal_N):
    str_list.append('[%d, %d],\n' %
                    (random.randint(0, normal_range-1), random.randint(0, normal_range-1)))

for _ in range(abnormal_N):
    str_list.append('[%d, %d],\n' %
                    (random.randint(normal_range, abnormal_range-1), random.randint(normal_range, abnormal_range-1)))


# for _ in range(normal_N):
#     str_list.append('[%d],\n' %
#                     (random.randint(0, normal_range-1)))

# for _ in range(abnormal_N):
#     str_list.append('[%d],\n' %
#                     (random.randint(normal_range, abnormal_range-1)))


random.shuffle(str_list)

print("".join(str_list))
