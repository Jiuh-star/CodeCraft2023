from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, List, Sequence, Union

AdjMatrix = List[List[float]]


class Price(NamedTuple):
    purchase: int
    sell: int


class WorkbenchType(NamedTuple):
    materials: List[int]
    cycle: int
    production: int


class Task(NamedTuple):
    src: Workbench
    dst: Workbench
    state: str


MAP_SCALE_FACTOR = 2
FLOAT_ZERO = 1E-4
MAX_EFFECTIVE_DISTANCE = 0.4  # m
STATIC_ROBOT_RADIUS = 0.45  # m
DYNAMIC_ROBOT_RADIUM = 0.53  # m
ROBOT_DENSITY = 20  # kg/m^2
MAX_FORWARD_VELOCITY = 6  # m/s
MAX_BACKWARD_VELOCITY = 2  # m/s
MAX_ROTATE_VELOCITY = math.pi  # rad/s
MAX_TRACTION = 250  # N
MAX_TORQUE = 50  # N*m

PRICE_LIST = [
    Price(purchase=0, sell=0),  # 占位
    Price(purchase=3000, sell=6000),  # 3,000
    Price(purchase=4400, sell=7600),  # 3,200
    Price(purchase=5800, sell=9200),  # 3,400
    Price(purchase=15400, sell=22500),  # 7,100
    Price(purchase=17200, sell=25000),  # 7,800
    Price(purchase=19200, sell=27500),  # 8,300
    Price(purchase=76000, sell=105000),  # 29,000
]

# 工作台需求表
WORKBENCH_TABLE = [
    WorkbenchType([], 0, 0),  # 占位
    WorkbenchType([], 50, 1),
    WorkbenchType([], 50, 2),
    WorkbenchType([], 50, 3),
    WorkbenchType([1, 2], 500, 4),
    WorkbenchType([1, 3], 500, 5),
    WorkbenchType([2, 3], 500, 6),
    WorkbenchType([4, 5, 6], 1000, 7),
    WorkbenchType([7], 1, 0),
    WorkbenchType([1, 2, 3, 4, 5, 6, 7], 1, 0),
]


@dataclass
class Workbench(object):
    id: int
    type: int
    position: List[float, float]
    remain_flame: int
    materials: List[int]
    ready_for_sell: bool

    def __post_init__(self):
        self.all_materials = WORKBENCH_TABLE[self.type].materials
        self.working_cycle = WORKBENCH_TABLE[self.type].cycle
        self.production = WORKBENCH_TABLE[self.type].production

    @property
    def needs(self):
        return [material for material in self.all_materials if material not in self.materials]

    def dist(self, other: Union[Robot, Workbench]):
        return get_distance(*self.position, *other.position)

    def parse_materials(self, digit: int):
        digit = bin(digit)[2:].rjust(8, "0")  # e.g. 00000110 -> [1, 2]
        self.materials = [i for i, bit in enumerate(reversed(digit)) if bit != '0']


@dataclass
class Robot(object):
    id: int
    in_workbench: int
    carry: int
    time_value_factor: float
    collision_value_factor: float
    angular_velocity: float
    velocity_vector: List[float, float]
    toward: float
    position: List[float, float]
    task: Union[Task, None] = None

    def _correct_robot_collision(self, others: List[Robot], glide_vector: List[float]):
        # 避让策略，使用碰撞盒进行检测
        factor = 1.1
        box_vector = [
            factor * (self.glide_vector[0] + self.radius),
            factor * (self.glide_vector[1] + self.radius),
        ]
        box = get_distance(*box_vector, 0, 0)
        other = min([(self.dist(other), other) for other in others], key=lambda x: x[0])
        if other[0] > 1.5 * box:
            return
        other = other[1]

        if self.id > other.id:
            self.forward(0)
        else:
            d = [
                other.position[0] - self.position[0],
                other.position[1] - self.position[1],
            ]
            v = self.velocity_vector
            dot = abs(d[0] * v[0] + d[1] * v[1])
            cross = v[0] * d[1] - v[1] * d[0]  # v × d 叉积
            d_norm = get_distance(*d, 0, 0)
            v_norm = get_distance(*v, 0, 0)
            alpha = math.acos(min(max(dot / (d_norm + v_norm), 1), 0))  # 速度矢量与连线的夹角
            theta = math.copysign(math.pi - alpha, - cross)
            if abs(theta) < math.pi / 180 * 15:
                self.rotate(theta)
            else:
                self.rotate(math.copysign(math.pi, theta))

    def _correct_border_collision(self, glide_vector: List[float]):
        # 碰到边界前停下，旋转后再前进
        # 可能撞墙，停下来，旋转后再前进
        factor = 1.2
        box_vector = [
            factor * (glide_vector[0] + self.radius),
            factor * (glide_vector[1] + self.radius)
        ]

        # 可能装西墙
        if self.position[0] < box_vector[0] and abs(self.toward) > math.pi / 2:
            self.forward(0)

        # 可能装东墙
        if self.position[0] > 50 - box_vector[0] - self.radius and abs(self.toward) < math.pi / 2:
            self.forward(0)

        # 可能撞南墙
        if self.position[1] < box_vector[1] and self.toward < 0:
            self.forward(0)

        # 可能撞北墙
        if self.position[1] > 50 - box_vector[1] and self.toward > 0:
            self.forward(0)

    @property
    def radius(self):
        return DYNAMIC_ROBOT_RADIUM if self.carry else STATIC_ROBOT_RADIUS

    @property
    def glide_vector(self):
        """ 滑动向量 """

        velocity = self.velocity_vector
        acceleration = MAX_TRACTION / (ROBOT_DENSITY * math.pi * (self.radius ** 2))  # a = F / m
        acceleration = [
            acceleration * math.cos(abs(self.toward)),
            acceleration * math.sin(abs(self.toward))
        ]
        # 滑行向量，即碰撞盒
        glide_vector = [
            (velocity[0] * velocity[0]) / (2 * acceleration[0] or 1),
            (velocity[1] * velocity[1]) / (2 * acceleration[1] or 1),
        ]

        return glide_vector

    def forward(self, velocity: float):
        print("forward", self.id, velocity)

    def rotate(self, velocity: float):
        print("rotate", self.id, velocity)

    def buy(self):
        print("buy", self.id)

    def sell(self):
        print("sell", self.id)

    def destroy(self):
        print("destroy", self.id)

    def dist(self, other: Union[Robot, Workbench]):
        return get_distance(*self.position, *other.position)

    def add_task(self, profit_matrix: List[List[float]], workbenches: List[Workbench]):
        arrows = [(weight, src, dst)
                  for src, row in enumerate(profit_matrix)
                  for dst, weight in enumerate(row)
                  if weight > FLOAT_ZERO]

        if arrows:
            arrows.sort(reverse=True)
            src, dst = workbenches[arrows[0][1]], workbenches[arrows[0][2]]
            self.task = Task(src, dst, state="buy")

    def goto(self, x: float, y: float, others: List[Robot]) -> float:

        # 计算角度，前进
        alpha = self.toward
        beta = math.atan2(self.position[1] - y, self.position[0] - x)
        theta = math.pi + beta - alpha
        theta = theta - 2 * math.pi if theta > math.pi else theta
        theta = theta + 2 * math.pi if theta < - math.pi else theta

        glide_vector = self.glide_vector
        dist = get_distance(*self.position, x, y)

        if abs(theta) < math.pi / 180 * 15:
            if dist > MAX_EFFECTIVE_DISTANCE + get_distance(*glide_vector, 0, 0):
                velocity = MAX_FORWARD_VELOCITY
            else:
                velocity = MAX_FORWARD_VELOCITY / 6

            self.rotate(theta)
            self.forward(velocity)
        else:
            self.rotate(math.copysign(math.pi, theta))
            self.forward(MAX_FORWARD_VELOCITY / 6)

        # 碰撞修正
        self._correct_robot_collision(others, glide_vector)
        self._correct_border_collision(glide_vector)

        return dist

    def go_on(self, others: List[Robot]):
        if self.task is None:
            return

        # 去 src 已经买了东西，要么出错，要么已经买好了
        if self.task.state == "buy" and self.carry:
            self.task = Task(self.task.src, self.task.dst, "sell")

        # 去 src 买东西
        elif self.task.state == "buy":
            dist = self.goto(*self.task.src.position, others=others)
            if dist < MAX_EFFECTIVE_DISTANCE:
                # 到了就买产品
                self.buy()

        # 去 dst 没带东西，要么出错，要么已经卖掉了
        if self.task.state == "sell" and not self.carry:
            self.task = None

        # 去 dst 卖东西
        elif self.task.state == "sell":
            dist = self.goto(*self.task.dst.position, others=others)
            if dist < MAX_EFFECTIVE_DISTANCE:
                # 到了就卖掉
                self.sell()


def parse_map(raw_map: Sequence[str], scale_factor: float):
    num_row = len(raw_map)

    robots: List[Robot] = []
    workbenches: List[Workbench] = []

    for reversed_row, line in enumerate(raw_map):
        row = num_row - reversed_row
        line = line.strip()

        for col, data in enumerate(line):
            if data.isdigit():
                workbenches.append(Workbench(
                    id=len(workbenches),
                    type=int(data),
                    position=[(row + 0.5) / scale_factor, (col + 0.5) / scale_factor],
                    remain_flame=-1,
                    materials=[],
                    ready_for_sell=False
                ))

            elif data.isalpha():
                robots.append(Robot(
                    id=len(robots),
                    in_workbench=-1,
                    carry=0,
                    time_value_factor=0,
                    collision_value_factor=0,
                    angular_velocity=0,
                    velocity_vector=[0, 0],
                    toward=0,
                    position=[(row + 0.5) / scale_factor, (col + 0.5) / scale_factor]
                ))

    return robots, workbenches


def get_distance(x1, y1, x2, y2):
    x = (x1 - x2)
    y = (y1 - y2)
    return math.sqrt(x * x + y * y)


def parse_profit_matrix(robot: Robot, workbenches: List[Workbench], other_robots: List[Robot]) -> AdjMatrix:
    profit_matrix = [[0.] * len(workbenches) for _ in workbenches]

    for row, src in enumerate(workbenches):
        for col, dst in enumerate(workbenches):
            # 如果 src 准备好销售，且能卖给 dst，且不是其它机器人的目标
            if any([other.task.dst.id == dst.id for other in other_robots]):
                continue

            if src.ready_for_sell and src.production in dst.needs:  # FIXME 并行
                dist_w = src.dist(dst)  # 工作台之间的距离
                dist_r = src.dist(robot)  # 机器人离 src 的距离
                t = (dist_w + dist_r) / 2  # 理论所需时间
                price = PRICE_LIST[src.production]
                profit = (price.sell - price.purchase) / t  # 单位时间利润

                # 利润修正，以反应比赛需求
                profit = max([len(dst.materials) + 1, 1.5]) * profit
                profit = [0, 1, 1, 1, 1.5, 1.5, 1.5, 2][src.type] * profit
                profit = (1 - sum([
                    0.1 for other in other_robots
                    if other.task.dst.dist(dst) < 5 or other.task.src.dist(src) < 5
                ])) * profit

                profit_matrix[row][col] = profit

    return profit_matrix


def read_map():
    raw_map = []
    while True:
        line = input()
        if line == "OK":
            break
        raw_map.append(line)

    robots, workbenches = parse_map(raw_map, MAP_SCALE_FACTOR)

    ok()

    return robots, workbenches


def start_read():
    try:
        line = input()
    except EOFError:
        exit(0)

    parts = line.split()
    frame_id, money = int(parts[0]), int(parts[1])

    return frame_id, money


def stop_read():
    input()  # ok


def read_workbenches(workbenches: List[Workbench]):
    num_workbench = int(input())

    for workbench in workbenches:
        parts = input().split()
        workbench.type = int(parts[0])
        workbench.position[0] = float(parts[1])
        workbench.position[1] = float(parts[2])
        workbench.remain_flame = int(parts[3])
        workbench.parse_materials(int(parts[4]))
        workbench.ready_for_sell = bool(int(parts[5]))


def read_robots(robots: List[Robot]):
    for robot in robots:
        parts = input().split()

        robot.in_workbench = int(parts[0])
        robot.carry = int(parts[1])
        robot.time_value_factor = float(parts[2])
        robot.collision_value_factor = float(parts[3])
        robot.angular_velocity = float(parts[4])
        robot.velocity_vector[0] = float(parts[5])
        robot.velocity_vector[1] = float(parts[6])
        robot.toward = float(parts[7])
        robot.position[0] = float(parts[8])
        robot.position[1] = float(parts[9])


def ok():
    print("OK", flush=True)


def start_operate(frame_id):
    print(frame_id)


def stop_operate():
    ok()


def debug(obj):
    import sys
    from pprint import pformat
    print(pformat(obj), file=sys.stderr, flush=True)


def main():
    robots, workbenches = read_map()

    while True:
        frame_id, money = start_read()
        read_workbenches(workbenches)
        read_robots(robots)
        stop_read()

        start_operate(frame_id)
        # 贪心策略，机器人优先级按序号排列
        for robot in robots:
            other_robots = [other for other in robots if other.id != robot.id]
            # 如果机器人有工作，让其继续完成工作（就是不销毁 ^-^）
            if robot.task:
                robot.go_on(other_robots)
                continue

            # 如果没有工作，则找一个工作
            other_busy_robots = [other for other in other_robots if other.task]
            profit_matrix = parse_profit_matrix(robot, workbenches, other_busy_robots)
            robot.add_task(profit_matrix, workbenches)

        stop_operate()


if __name__ == '__main__':
    main()
