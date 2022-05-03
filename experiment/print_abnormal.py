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

service_changes = [
    # normal changes
    ('ts-route-service', 'cqqcqq/route_inv_contacts:latest'),
    ('ts-order-service', 'cqqcqq/order_inv_contacts:latest'),
    ('ts-auth-service', 'cqqcqq/auth_inv_order:latest'),

    # abnormal changes
    ('ts-ticketinfo-service', 'cqqcqq/ticketinfo_oom:latest'),
    ('ts-travel-service', 'cqqcqq/travel_oom:latest'),

    ('ts-route-service', 'cqqcqq/route_sleep:latest'),
    ('ts-order-service', 'cqqcqq/order_sleep:latest'),
    ('ts-auth-service', 'cqqcqq/auth_sleep:latest'),

    ('ts-order-service', 'cqqcqq/order_port:latest'),
    ('ts-route-service', 'cqqcqq/route_port:latest'),
    ('ts-user-service', 'cqqcqq/user_port:latest'),

    ('ts-order-service', 'cqqcqq/order_table:latest'),
    ('ts-route-service', 'cqqcqq/route_table:latest'),
    ('ts-user-service', 'cqqcqq/user_table:latest')
]

change_order1 = [
    [0], [3], [6], [5], [7],
    [1], [7], [2], [0], [3],
    [11], [2], [5], [10], [8],
    [5], [2], [6], [5], [2],
    [1], [8], [9], [13], [11],
    [0], [1], [12], [0], [1],
    [2], [3], [1], [6], [9],
    [10], [4], [3], [8], [3],
    [11], [8], [0], [2], [8],
    [2], [0], [5], [6], [0],
]

change_order2 = [
    [3, 4], [10, 12], [12, 7], [3, 10], [10, 11],
    [9, 9], [5, 10], [2, 1], [9, 3], [6, 11],
    [8, 13], [13, 7], [8, 11], [11, 7], [2, 0],
    [0, 1], [10, 8], [2, 0], [13, 8], [1, 0],
    [2, 0], [0, 2], [8, 4], [13, 6], [2, 1],
    [0, 1], [10, 8], [12, 5], [3, 11], [2, 0],
    [1, 2], [5, 12], [6, 4], [6, 10], [0, 1],
    [11, 13], [1, 0], [7, 6], [1, 1], [2, 1],
    [2, 0], [1, 2], [6, 10], [12, 5], [1, 2],
    [4, 5], [8, 12], [0, 1], [13, 9], [0, 2],
]

print('| 序号 | abnormal 1 | abnormal 2 | change 1 | change 2 |')
print('| --- | --- | --- | --- | --- |')
for i in range(50):
    print(f'| {i} | {fault[random_fault[i]]} | {fault[double_random_fault[i][0]]}, {fault[double_random_fault[i][1]]} | {service_changes[change_order1[i][0]][1]} | {service_changes[change_order2[i][0]][1]}, {service_changes[change_order2[i][1]][1]} |')