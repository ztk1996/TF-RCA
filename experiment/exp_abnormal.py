import os
from typing import Callable
from autoquery.queries import Query
from autoquery.scenarios import *
from autoquery.utils import random_from_weighted
import logging
import random
import time
import argparse
from multiprocessing import Process, Pool
from query import *

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("autoquery-manager")

url = 'http://139.196.152.44:32677'
minute = 60
hour = 60 * minute

# 记录异常情况下函数执行时间
fun_dic = {}
timeout = 5 * minute


def constant_query(targets: list):
    start = time.time()
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    def payment_scenario():
        query_and_pay(q)

    def cancel_scenario():
        query_and_cancel(q)

    def collect_scenario():
        query_and_collect(q)

    def execute_scenario():
        query_and_execute(q)

    if not q.login():
        return

    count = random.randint(3, 5)
    interval = random.randint(3, 5)

    query_weights = {
        q.query_cheapest: 20,
        q.query_orders: 30,
        q.query_food: 5,
        q.query_high_speed_ticket: 50,
        q.query_contacts: 10,
        q.query_min_station: 20,
        q.query_quickest: 20,
        q.query_high_speed_ticket_parallel: 10,
        preserve_scenario: 30,
        payment_scenario: 20,
        cancel_scenario: 20,
        collect_scenario: 20,
        execute_scenario: 20,
    }

    forbid_query = {
        "travel": [q.query_high_speed_ticket,
                   q.query_high_speed_ticket_parallel,
                   q.query_min_station,
                   q.query_cheapest,
                   q.query_quickest,
                   preserve_scenario],
        "ticketinfo": [q.query_high_speed_ticket,
                       q.query_high_speed_ticket_parallel,
                       q.query_min_station,
                       q.query_cheapest,
                       q.query_quickest,
                       preserve_scenario],
        "route": [q.query_high_speed_ticket,
                  q.query_min_station,
                  q.query_cheapest,
                  q.query_quickest,
                  preserve_scenario],
        "order": [q.query_high_speed_ticket,
                  q.query_min_station,
                  q.query_cheapest,
                  q.query_quickest,
                  q.query_orders,
                  payment_scenario,
                  cancel_scenario,
                  collect_scenario],
        "basic": [q.query_high_speed_ticket,
                  q.query_min_station,
                  q.query_cheapest,
                  q.query_quickest],
        "user": [cancel_scenario],
        "travel-plan": [q.query_min_station,
                        q.query_cheapest,
                        q.query_quickest],
        "station": [q.query_high_speed_ticket,
                    q.query_high_speed_ticket_parallel,
                    q.query_min_station,
                    q.query_cheapest,
                    q.query_quickest,
                    q.query_orders,
                    q.query_other_orders],

    }

    for target in targets:
        if fun in forbid_query.keys():
            for fun in forbid_query[target]:
                if fun in query_weights:
                    del query_weights[fun]

    for _ in range(0, count):
        if time.time() - start > timeout:
            return
        func = random_from_weighted(query_weights)
        logger.info(f'constant execute query: {func.__name__}')
        try:
            func()
        except Exception:
            logger.exception(
                f'constant query {func.__name__} got an exception')

        time.sleep(interval)

    return


chaos_path = {
    'basic-network-delay': 'chaos/network_delay/basic_network_delay.yml',
    'order-network-delay': 'chaos/network_delay/order_network_delay.yml',
    'route-network-delay': 'chaos/network_delay/route_network_delay.yml',
    'station-network-delay': 'chaos/network_delay/station_network_delay.yml',
    'ticketinfo-network-delay': 'chaos/network_delay/ticketinfo_network_delay.yml',
    'travel-network-delay': 'chaos/network_delay/travel_network_delay.yml',
    'travel-plan-network-delay': 'chaos/network_delay/travel_plan_network_delay.yml',
    'user-network-delay': 'chaos/network_delay/user_network_delay.yml',
    'config-network-delay': 'chaos/network_delay/config.yml',
    'consign-network-delay': 'chaos/network_delay/consign.yml',
    'order-other-network-delay': 'chaos/network_delay/order_other.yml',
    'price-network-delay': 'chaos/network_delay/price.yml',
    'rebook-network-delay': 'chaos/network_delay/rebook.yml',
    'seat-network-delay': 'chaos/network_delay/seat.yml',
    'train-network-delay': 'chaos/network_delay/train.yml',

    'basic-cpu-stress': 'chaos/cpu_stress/basic_stress_cpu.yml',
    'order-cpu-stress': 'chaos/cpu_stress/order_stress_cpu.yml',
    'route-cpu-stress': 'chaos/cpu_stress/route_stress_cpu.yml',
    'station-cpu-stress': 'chaos/cpu_stress/station_stress_cpu.yml',
    'ticketinfo-cpu-stress': 'chaos/cpu_stress/ticketinfo_stress_cpu.yml',
    'travel-cpu-stress': 'chaos/cpu_stress/travel_stress_cpu.yml',
    'travel-plan-cpu-stress': 'chaos/cpu_stress/travel_plan_stress_cpu.yml',
    'user-cpu-stress': 'chaos/cpu_stress/user_stress_cpu.yml',

    'basic-http-outbound': 'chaos/http_outbound/basic_http_outbound.yml',
    'order-http-outbound': 'chaos/http_outbound/order_http_outbound.yml',
    'route-http-outbound': 'chaos/http_outbound/route_http_outbound.yml',
    'station-http-outbound': 'chaos/http_outbound/station_http_outbound.yml',
    'ticketinfo-http-outbound': 'chaos/http_outbound/ticketinfo_http_outbound.yml',
    'travel-http-outbound': 'chaos/http_outbound/travel_http_outbound.yml',
    'travel-plan-http-outbound': 'chaos/http_outbound/travel_plan_http_outbound.yml',
    'user-http-outbound': 'chaos/http_outbound/user_http_outbound.yml',
    'config-http-outbound': 'chaos/http_outbound/config.yml',
    'consign-http-outbound': 'chaos/http_outbound/consign.yml',
    'order-other-http-outbound': 'chaos/http_outbound/order_other.yml',
    'price-http-outbound': 'chaos/http_outbound/price.yml',
    'rebook-http-outbound': 'chaos/http_outbound/rebook.yml',
    'seat-http-outbound': 'chaos/http_outbound/seat.yml',
    'train-http-outbound': 'chaos/http_outbound/train.yml',

    'basic-http-code': 'chaos/http_code/basic.yml',
    'order-http-code': 'chaos/http_code/order.yml',
    'route-http-code': 'chaos/http_code/route.yml',
    'station-http-code': 'chaos/http_code/station.yml',
    'ticketinfo-http-code': 'chaos/http_code/ticketinfo.yml',
    'travel-http-code': 'chaos/http_code/travel.yml',
    'travel-plan-http-code': 'chaos/http_code/travel_plan.yml',
    'user-http-code': 'chaos/http_code/user.yml',
    'config-http-code': 'chaos/http_code/config.yml',
    'consign-http-code': 'chaos/http_code/consign.yml',
    'order-other-http-code': 'chaos/http_code/order_other.yml',
    'price-http-code': 'chaos/http_code/price.yml',
    'rebook-http-code': 'chaos/http_code/rebook.yml',
    'seat-http-code': 'chaos/http_code/seat.yml',
    'train-http-code': 'chaos/http_code/train.yml',
}


def apply(file_path):
    command = "kubectl apply -f " + file_path
    os.system(command)


def delete(file_path):
    command = "kubectl delete -f " + file_path
    os.system(command)


def select_fault(idx: int, module: int = 1) -> list:
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
        'travel-http-outbound', 'travel-plan-http-outbound', 'user-http-outbound', 'config-http-outbound', 'consign-http-outbound',
        'order-other-http-outbound', 'price-http-outbound', 'rebook-http-outbound', 'seat-http-outbound', 'train-http-outbound',

        # 'basic-cpu-stress', 'order-cpu-stress', 'route-cpu-stress', 'station-cpu-stress', 'ticketinfo-cpu-stress',
        # 'travel-cpu-stress', 'travel-plan-cpu-stress', 'user-cpu-stress',
    ]
    random_fault = [13, 30, 34, 25, 9, 11, 35, 21, 20, 44, 38, 16, 32, 29, 22, 17, 19, 26, 0, 37,
                    10, 41, 40, 6, 14, 8, 27, 24, 31, 2, 12, 3, 4, 5, 43, 42, 33, 23, 39, 18, 1, 15,
                    36, 28, 7, 0, 15, 30, 1, 16]
    double_random_fault = [[35, 38], [2, 16], [5, 9], [26, 30], [29, 15], [9, 15], [1, 0], [41, 39], [15, 1], [13, 0], [22, 27], [10, 41], [23, 14], [4, 12], [17, 23], [33, 26], [6, 37], [37, 16], [25, 36], [17, 8], [39, 14], [7, 10], [44, 18], [33, 24], [
        40, 35], [28, 20], [30, 31], [5, 20], [16, 18], [38, 42], [21, 12], [1, 29], [3, 42], [34, 2], [22, 25], [13, 6], [3, 44], [43, 19], [24, 27], [11, 4], [0, 0], [19, 8], [34, 1], [21, 43], [32, 36], [28, 40], [31, 30], [11, 7], [32, 30], [16, 15]]

    if idx < 0 or idx > 49:
        return []
    if module == 2:
        return [fault[double_random_fault[idx][0]], fault[double_random_fault[idx][1]]]
    return [fault[random_fault[idx]]]


def workflow(times: int = 50, task_timeout: int = 5 * minute, module: int = 1):
    # task for each query
    tasks = {
        "travel": query_travel,
        "ticketinfo": query_ticketinfo,
        "route": query_route,
        "order": query_order,
        "basic": query_basic,
        "user": query_user,
        "travel-plan": query_travel_plan,
        "station": query_station,
        "config": query_config,
        "consign": query_consign,
        "order-other": query_order_other,
        "price": query_price,
        "rebook": query_rebook,
        "seat": query_seat,
        "train": query_train,
    }
    # # 持续进行正常query
    # logger.info('start constant query')
    # p.apply_async(constant_query)
    request_period_log = []
    for current in range(times):
        # 选择故障
        faults = select_fault(current, module)
        if len(faults) == 0:
            logger.info("no task, waiting for 1 minute")
            time.sleep(1 * minute)
            continue
        # 选择task
        targets = []
        task_list = []
        for fault in faults:
            fault_split = fault.split("-")
            target = fault_split[0]
            if fault_split[0] == "travel" and fault_split[1] == "plan":
                target = "travel-plan"
            elif fault_split[0] == "order" and fault_split[1] == 'other':
                target = "order-other"
            task = tasks[target]
            targets.append(target)
            task_list.append(task)
        # 注入故障
        for fault in faults:
            logger.info(f'fault inject: {fault}')
            apply(chaos_path[fault])
        time.sleep(10)
        # 正常
        p = Pool(15)
        p.apply_async(constant_query, args=(targets))
        # 异常
        start_time = time.time()
        name_list = []
        for index, task in enumerate(task_list):
            logger.info(f'execute task: {task.__name__}')
            p.apply_async(task)
            p.apply_async(task)
            p.apply_async(task)
            p.apply_async(task)
            p.apply_async(task)
            name = "ts-" + targets[index] + "-service"
            name_list.append(name)

        # 恢复故障
        p.close()
        p.join()

        for fault in faults:
            logger.info(f'fault recover: {fault}')
            delete(chaos_path[fault])

        end_time = time.time()
        request_period_log.append(
            (name_list, int(round(start_time*1000)), int(round(end_time*1000))))
        # 间隔3min
        time.sleep(3 * minute)

    logger.info('waiting for constant query end...')
    print(f'request_period_log: {request_period_log}')
    return


def arguments():
    parser = argparse.ArgumentParser(description="query manager arguments")
    parser.add_argument(
        '--duration', help='query constant duration (times)', default=50)
    parser.add_argument('--url', help='train ticket server url',
                        default='http://139.196.152.44:32677')
    parser.add_argument('--module', help='single or double',
                        default=2)
    return parser.parse_args()


def main():
    args = arguments()
    global url
    url = args.url
    duration = int(args.duration)
    logger.info(f'start auto-query manager for {duration} times')
    module = int(args.module)
    logger.info('start query workflow')
    workflow(times=duration, module=module)
    logger.info('workflow ended')
    logger.info('auto-query manager ended')
    logger.info(f'execute_func_dict: {fun_dic}')


if __name__ == '__main__':
    main()
