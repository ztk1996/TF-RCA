from exp_structure_change import service_changes, change_order1, change_order2

# All fault
fault = [
    # 0-14
    'basic-network-delay', 'order-network-delay', 'route-network-delay', 'station-network-delay', 'ticketinfo-network-delay',
    'travel-network-delay', 'travel-plan-network-delay', 'user-network-delay', 'config-network-delay', 'consign-network-delay',
    'order-other-network-delay', 'price-network-delay', 'rebook-network-delay', 'seat-network-delay', 'train-network-delay',

    # 15-29
    'basic-http-code', 'order-http-code', 'route-http-code', 'station-http-code', 'ticketinfo-http-code',
    'travel-http-code', 'route-http-code', 'user-http-code', 'config-http-code', 'consign-http-code',
    'order-other-http-code', 'price-http-code', 'rebook-http-code', 'seat-http-code', 'train-http-code',

    # 30-44
    'basic-http-outbound', 'order-http-outbound', 'route-http-outbound', 'station-http-outbound', 'ticketinfo-http-outbound',
    'travel-http-outbound', 'travel-plan-http-outbound', 'user-http-outbound', 'config-network-delay', 'consign-network-delay',
    'order-other-network-delay', 'price-network-delay', 'rebook-network-delay', 'seat-network-delay', 'train-network-delay',

    # 'basic-cpu-stress', 'order-cpu-stress', 'route-cpu-stress', 'station-cpu-stress', 'ticketinfo-cpu-stress',
    # 'travel-cpu-stress', 'travel-plan-cpu-stress', 'user-cpu-stress',
]
random_fault = [13, 30, 34, 25, 9, 11, 35, 21, 20, 44, 38, 16, 32, 29, 22, 17, 19, 26, 0, 37,
                10, 41, 40, 6, 14, 8, 27, 24, 31, 2, 12, 3, 4, 5, 43, 42, 33, 23, 39, 18, 1, 15,
                36, 28, 7, 0, 15, 30, 1, 16]
double_random_fault = [[35, 38], [2, 16], [5, 9], [26, 30], [29, 15], [9, 15], [1, 0], [41, 39], [15, 1], [13, 0], [22, 27], [10, 41], [23, 14], [4, 12], [17, 23], [33, 26], [6, 37], [37, 16], [25, 36], [17, 8], [39, 14], [7, 10], [44, 18], [33, 24], [
    40, 35], [28, 20], [30, 31], [5, 20], [16, 18], [38, 42], [21, 12], [1, 29], [3, 42], [34, 2], [22, 25], [13, 6], [3, 44], [43, 19], [24, 27], [11, 4], [0, 0], [19, 8], [34, 1], [21, 43], [32, 36], [28, 40], [31, 30], [11, 7], [32, 30], [16, 15]]


print('| 序号 | abnormal 1 | abnormal 2 | change 1 | change 2 |')
print('| --- | --- | --- | --- | --- |')
for i in range(50):
    print(f'| {i} | {fault[random_fault[i]]} | {fault[double_random_fault[i][0]]}, {fault[double_random_fault[i][1]]} | {service_changes[change_order1[i][0]][1]} | {service_changes[change_order2[i][0]][1]}, {service_changes[change_order2[i][1]][1]} |')
