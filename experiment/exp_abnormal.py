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
    interval = random.randint(10, 20)

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


chaos_path = {'basic-network-delay': 'chaos/network_delay/basic_network_delay.yml',
              'order-network-delay': 'chaos/network_delay/order_network_delay.yml',
              'route-network-delay': 'chaos/network_delay/route_network_delay.yml',
              'station-network-delay': 'chaos/network_delay/station_network_delay.yml',
              'ticketinfo-network-delay': 'chaos/network_delay/ticketinfo_network_delay.yml',
              'travel-network-delay': 'chaos/network_delay/travel_network_delay.yml',
              'travel-plan-network-delay': 'chaos/network_delay/travel_plan_network_delay.yml',
              'user-network-delay': 'chaos/network_delay/user_network_delay.yml',

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
              'user-http-outbound': 'chaos/http_outbound/user_http_outbound.yml'}


def apply(file_path):
    command = "kubectl apply -f " + file_path
    os.system(command)


def delete(file_path):
    command = "kubectl delete -f " + file_path
    os.system(command)


def select_fault(idx: int, module: int = 1) -> list:
    # All fault
    fault = [
        # 0-7
        'basic-network-delay', 'order-network-delay', 'route-network-delay', 'station-network-delay',
        'ticketinfo-network-delay', 'travel-network-delay', 'travel-plan-network-delay', 'user-network-delay',
        # 8-15
        'basic-cpu-stress', 'order-cpu-stress', 'route-cpu-stress', 'station-cpu-stress', 'ticketinfo-cpu-stress',
        'travel-cpu-stress', 'travel-plan-cpu-stress', 'user-cpu-stress', 'basic-http-outbound',
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
        [6, 14], [13, 16], [7, 15], [14, 22], [21, 7], [11, 11], [
            0, 19], [19, 10], [1, 18], [22, 0], [8, 17], [18, 6],

        [19, 7], [14, 6]]
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
            task = tasks[target]
            targets.append(target)
            task_list.append(task)
        # 注入故障
        for fault in faults:
            logger.info(f'fault inject: {fault}')
            apply(chaos_path[fault])
        time.sleep(10)
        # 正常
        p = Pool(4)
        p.apply_async(constant_query, args=(targets))
        # 异常
        start_time = time.time()
        name_list = []
        for index, task in enumerate(task_list):
            logger.info(f'execute task: {task.__name__}')
            p.apply_async(task)
            name = "ts-" + targets[index] + "-service"
            name_list.append(name)

        # 恢复故障
        p.close()
        p.join()
        end_time = time.time()
        request_period_log.append(
            (name_list, int(round(start_time*1000)), int(round(end_time*1000))))

        for fault in faults:
            logger.info(f'fault recover: {fault}')
            delete(chaos_path[fault])
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
