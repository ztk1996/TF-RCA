from autoquery.queries import Query
from autoquery.scenarios import *
from autoquery.utils import random_from_weighted
import time
from typing import Callable

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger("autoquery-manager")

url = 'http://139.196.152.44:32677'
minute = 60
hour = 60*minute
timeout = 5*minute


def random_query(q: Query, weights: dict, count: int = random.randint(1, 3), inteval: int = random.randint(1, 3)):
    """
    登陆一个用户并按权重随机发起请求
    :param weights: 权重dict
    :param count: 请求次数
    :param inteval: 请求间隔
    """
    random.seed()
    try:
        if not q.login():
            return
    except Exception:
        logger.exception(f'login got an exception')
        return

    for _ in range(0, count):
        func = random_from_weighted(weights)
        logger.info(f'execure query: {func.__name__}')
        try:
            func()
        except Exception:
            logger.exception(f'query {func.__name__} got an exception')

        time.sleep(inteval)

    return


def run(task: Callable, timeout: int):
    start = time.time()
    while time.time() - start < timeout:
        task()
        time.sleep(1)
    return


def query_route():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_food: 10,
        q.query_normal_ticket: 10,

        q.query_route: 50,
        q.query_high_speed_ticket: 50,
        preserve_scenario: 20,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_order():
    q = Query(url)

    def payment_scenario():
        query_and_pay(q)

    def cancel_scenario():
        query_and_cancel(q)

    def collect_scenario():
        query_and_collect(q)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_food: 10,
        q.query_normal_ticket: 10,
        q.query_route: 10,

        q.query_high_speed_ticket: 20,
        q.query_orders: 30,
        payment_scenario: 30,
        cancel_scenario: 20,
        collect_scenario: 20,
        preserve_scenario: 20,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_auth():
    q = Query(url)
    query_weights = {
        q.query_food: 10,
        q.query_normal_ticket: 10,
        q.query_route: 10,
        q.query_high_speed_ticket: 10,
        q.login: 20
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_ticketinfo():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_route: 10,

        q.query_high_speed_ticket: 20,
        q.query_min_station: 20,
        q.query_cheapest: 20,
        q.query_quickest: 20,
        preserve_scenario: 30,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_travel():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_food: 10,
        q.query_normal_ticket: 10,
        q.query_route: 10,

        q.query_high_speed_ticket: 20,
        q.query_min_station: 20,
        q.query_cheapest: 20,
        q.query_quickest: 20,
        preserve_scenario: 20,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_user():
    q = Query(url)

    def cancel_scenario():
        query_and_cancel(q)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_high_speed_ticket: 10,

        preserve_scenario: 50,
        cancel_scenario: 50,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return


def query_basic():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_travel_plan():
    q = Query(url)

    query_weights = {
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_station():
    q = Query(url)

    query_weights = {
        q.query_high_speed_ticket: 10,
        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,
        q.query_orders: 10,
        q.query_other_orders: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_config():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_route: 10,

        q.query_admin_basic_config: 20,

        q.query_min_station: 10,
        q.query_quickest: 10,
        q.query_cheapest: 10,

        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_consign():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    def consign_scenario():
        query_and_consign(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_route: 10,

        consign_scenario: 30,
        preserve_scenario: 30,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_order_other():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_advanced_ticket: 10,
        q.query_route: 10,

        q.query_high_speed_ticket: 30,
        q.query_other_orders: 30,

        preserve_scenario: 30
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_price():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_advanced_ticket: 10,

        q.query_admin_basic_price: 10,
        q.query_normal_ticket: 10,
        q.query_high_speed_ticket: 10,
        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_rebook():
    q = Query(url)

    def rebook_scenario():
        query_and_rebook(q)

    query_weights = {
        q.query_normal_ticket: 10,
        q.query_advanced_ticket: 10,
        q.query_high_speed_ticket: 10,

        rebook_scenario: 40,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_seat():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_orders: 10,
        q.query_route: 10,

        q.query_min_station: 10,
        q.query_cheapest: 10,
        q.query_quickest: 10,

        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_train():
    q = Query(url)

    query_weights = {
        q.query_route: 10,
        q.query_other_orders: 10,

        q.query_normal_ticket: 10,
        q.query_high_speed_ticket: 10,

        q.query_min_station: 10,
        q.query_quickest: 10,
        q.query_cheapest: 10,
    }

    def task():
        random_query(q, query_weights)

    return run(task, timeout)


def query_food():
    q = Query(url)

    def preserve_scenario():
        query_and_preserve(q)

    query_weights = {
        q.query_food: 50,

        q.query_route: 10,
        q.query_high_speed_ticket: 10,
        q.query_cheapest: 10,
        preserve_scenario: 10,
    }

    def task():
        random_query(q, query_weights)

    run(task, timeout)
    return
