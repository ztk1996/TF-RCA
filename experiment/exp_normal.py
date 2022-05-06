from ast import Call
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
hour = 60*minute


def constant_query(timeout: int = 24*hour):
    start = time.time()
    q = Query(url)
    random.seed()
    
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

    while time.time()-start < timeout:
        query_num = random.randint(2, 10)
        new_login = random_from_weighted({True: 70, False: 30})
        if new_login or q.token == "":
            while not q.login():
                time.sleep(10)
                continue

        query_weights = {
            q.query_cheapest: 10,
            q.query_orders: 10,
            q.query_food: 5,
            q.query_high_speed_ticket: 10,
            q.query_contacts: 10,
            q.query_min_station: 10,
            q.query_quickest: 10,
            q.query_high_speed_ticket_parallel: 10,

            preserve_scenario: 10,
            payment_scenario: 10,
            cancel_scenario: 10,
            collect_scenario: 10,
            execute_scenario: 10,
        }

        for i in range(0, query_num):
            func = random_from_weighted(query_weights)
            logger.info(f'execure constant query: {func.__name__}')
            try:
                func()
            except Exception:
                logger.exception(f'query {func.__name__} got an exception')

            time.sleep(random.randint(1, 3))

    return


def select_task(n: int) -> List[Callable]:
    task_list = [
        query_route, query_order, query_auth, query_ticketinfo, query_travel,
        query_user, query_basic, query_travel, query_travel_plan, query_station,
        query_config, query_consign, query_order_other, query_price, query_rebook,
        query_seat, query_train,
    ]

    return random.sample(task_list, n)


def workflow(timeout: int = 24*hour, task_timeout: int = 5*minute):
    start = time.time()

    p = Pool(10)
    logger.info('start constant query')
    p.apply_async(constant_query)

    while time.time() - start < timeout:
        tasks = select_task(3)

        for task in tasks:
            logger.info(f'execute task: {task.__name__}')
            p.apply_async(task)

        time.sleep(task_timeout)

    p.close()
    logger.info('waiting for constant query end...')
    p.join()
    return


def arguments():
    parser = argparse.ArgumentParser(description="query manager arguments")
    parser.add_argument(
        '--duration', help='query constant duration (hour)', default=100)
    parser.add_argument('--url', help='train ticket server url',
                        default='http://139.196.152.44:32677')
    return parser.parse_args()


def main():
    args = arguments()
    global url
    url = args.url
    duration = int(args.duration) * hour
    logger.info(f'start auto-query manager for {duration//hour} hour(s)')

    logger.info('start query workflow')
    workflow(duration)
    logger.info('workflow ended')

    logger.info('auto-query manager ended')


if __name__ == '__main__':
    main()
