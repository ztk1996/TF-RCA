fault = [
	# 0-7
	'basic-network-delay', 'order-network-delay', 'route-network-delay', 'station-network-delay',
	'ticketinfo-network-delay', 'travel-network-delay', 'travel-plan-network-delay', 'user-network-delay',
	# 8-15
	'basic-network-delay', 'order-network-delay', 'route-network-delay', 'station-network-delay',
	'ticketinfo-network-delay', 'travel-network-delay', 'route-network-delay', 'user-network-delay',
	# 'basic-cpu-stress', 'order-cpu-stress', 'route-cpu-stress', 'station-cpu-stress', 'ticketinfo-cpu-stress',
	# 'travel-cpu-stress', 'travel-plan-cpu-stress', 'user-cpu-stress',
	'basic-http-outbound',
	# 16-23
	'order-http-outbound', 'route-http-outbound', 'station-http-outbound', 'ticketinfo-http-outbound',
	'travel-http-outbound', 'travel-plan-http-outbound', 'user-http-outbound']
random_fault = [10, 5, 13, 21, 11, 16, 4, 23, 1, 9, 17, 8, 18, 7, 19, 12, 6, 22, 15, 0, 20, 3, 14, 2,
				21, 10, 16, 5, 23, 4, 13, 1, 11, 7, 9, 6, 17, 8, 0, 18, 3, 12, 19, 2, 22, 15, 20, 14,
				1, 21]
double_random_fault = [
	[11, 14], [22, 20], [6, 15], [18, 23], [9, 10], [21, 4], [
		14, 11], [3, 2], [15, 9], [1, 0], [12, 22], [20, 13],
	[13, 16], [2, 1], [10, 12], [16, 18], [4, 5], [8, 8], [
		5, 17], [23, 3], [0, 21], [19, 7], [7, 19], [17, 6],

	[12, 13], [23, 4], [3, 12], [20, 5], [10, 9], [2, 23], [
		17, 3], [4, 8], [9, 2], [16, 20], [5, 1], [15, 21],
	[6, 14], [13, 16], [7, 15], [14, 22], [21, 7], [12, 11], [
		0, 19], [19, 10], [1, 18], [22, 0], [8, 17], [18, 6],
	[19, 7], [14, 6]]

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