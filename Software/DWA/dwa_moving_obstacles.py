import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

@dataclass
class Config:
    # 速度限制
    min_v: float = 0.0
    max_v: float = 8.8
    max_omega: float = 3.2

    # 加速度限制：本例假设加速、减速对称
    max_accel: float = 5.5
    max_angular_accel: float = 5.0

    # 制动减速度大小，必须为正
    brake_accel: float = 5.5
    brake_angular_accel: float = 5.0

    # 三个不同的时间参数
    control_dt: float = 0.1      # 多久重新选择一次速度
    prediction_time: float = 2.0
    simulation_dt: float = 0.05 # 预测与碰撞检查的分段长度

    # 候选速度采样间隔
    v_resolution: float = 0.05
    omega_resolution: float = 0.1

    robot_radius: float = 0.25
    safety_margin: float = 0.05

    distance_cap: float = 3.0
    goal_tolerance: float = 0.25

    heading_weight: float = 0.6
    distance_weight: float = 0.2
    speed_weight: float = 0.2


def move(x, y, theta, v, omega, dt):
    """恒定速度运动的解析解。"""
    if dt < 0:
        raise ValueError("dt 不能小于 0")

    new_theta = theta + omega * dt

    if abs(omega) < 1e-8:
        new_x = x + v * math.cos(theta) * dt
        new_y = y + v * math.sin(theta) * dt
    else:
        radius = v / omega
        new_x = x + radius * (
            math.sin(new_theta) - math.sin(theta)
        )
        new_y = y + radius * (
            math.cos(theta) - math.cos(new_theta)
        )

    return new_x, new_y, new_theta


def predict_trajectory(pose, v, omega, duration, dt):
    """返回包括初始位姿在内的轨迹点。"""
    if duration < 0 or dt <= 0:
        raise ValueError("duration 必须非负，dt 必须为正")

    trajectory = [tuple(pose)]
    elapsed = 0.0

    while duration - elapsed > 1e-10:
        step = min(dt, duration - elapsed)
        pose = move(*pose, v, omega, step)
        trajectory.append(pose)
        elapsed += step

    return trajectory


def calculate_dynamic_window(state, config):
    """state = (x, y, theta, 当前v, 当前omega)。"""
    _, _, _, current_v, current_omega = state

    # 动态窗口与硬件速度范围取交集
    v_lower = max(
        config.min_v,
        current_v - config.max_accel * config.control_dt,
    )
    v_upper = min(
        config.max_v,
        current_v + config.max_accel * config.control_dt,
    )

    omega_lower = max(
        -config.max_omega,
        current_omega
        - config.max_angular_accel * config.control_dt,
    )
    omega_upper = min(
        config.max_omega,
        current_omega
        + config.max_angular_accel * config.control_dt,
    )

    return v_lower, v_upper, omega_lower, omega_upper


def sample_axis(lower, upper, resolution):
    """包括上下界，并在范围允许时加入零。"""
    if resolution <= 0:
        raise ValueError("采样间隔必须为正")
    if lower > upper:
        return []

    count = int(math.floor((upper - lower) / resolution))
    values = [
        lower + i * resolution
        for i in range(count + 1)
    ]
    values.append(upper)

    if lower <= 0 <= upper:
        values.append(0.0)

    return sorted(set(values))


def sample_velocities(window, config):
    v_lower, v_upper, omega_lower, omega_upper = window

    for v in sample_axis(
        v_lower, v_upper, config.v_resolution
    ):
        for omega in sample_axis(
            omega_lower,
            omega_upper,
            config.omega_resolution,
        ):
            yield v, omega


def pose_collides(pose, obstacles, config):
    """障碍物格式：(中心x, 中心y, 半径)。"""
    x, y, _ = pose

    for ox, oy, obstacle_radius in obstacles:
        required_distance = (
            config.robot_radius
            + config.safety_margin
            + obstacle_radius
        )
        if math.hypot(x - ox, y - oy) <= required_distance:
            return True

    return False


def segment_collides(pose, v, omega, dt, obstacles, config):
    """
    保守检查整段运动，避免只检查离散端点而漏掉障碍。

    中间时刻的位置到该段任意位置的距离，
    不超过该段行驶长度的一半。
    因此用这个长度扩大检查圆。
    """
    middle = move(*pose, v, omega, dt / 2)
    extra_radius = abs(v) * dt / 2

    for ox, oy, obstacle_radius in obstacles:
        required_distance = (
            config.robot_radius
            + config.safety_margin
            + obstacle_radius
            + extra_radius
        )

        if math.hypot(
            middle[0] - ox, middle[1] - oy
        ) <= required_distance:
            return True

    return False


def trajectory_is_clear(pose, v, omega, duration,
                        obstacles, config):
    elapsed = 0.0

    while duration - elapsed > 1e-10:
        step = min(
            config.simulation_dt, duration - elapsed
        )

        if segment_collides(
            pose, v, omega, step, obstacles, config
        ):
            return False

        pose = move(*pose, v, omega, step)
        elapsed += step

    return True


def is_admissible(state, v, omega, obstacles, config):
    """
    检查：
    1. 按候选速度执行一个控制周期；
    2. 随后沿固定曲率制动直到停止。

    候选速度在周期开始时直接用于预测，
    这是本教学模型的近似。
    """
    pose = tuple(state[:3])

    if pose_collides(pose, obstacles, config):
        return False

    # 先检查下一个控制周期
    if not trajectory_is_clear(
        pose, v, omega, config.control_dt,
        obstacles, config,
    ):
        return False

    pose = move(
        *pose, v, omega, config.control_dt
    )

    # 选择共同停车时间，使两种减速度均不超限。
    stop_time = max(
        abs(v) / config.brake_accel,
        abs(omega) / config.brake_angular_accel,
    )

    if stop_time < 1e-10:
        return True

    # v、omega 按同一比例线性减小，保持曲率。
    elapsed = 0.0

    while stop_time - elapsed > 1e-10:
        step = min(
            config.simulation_dt, stop_time - elapsed
        )

        # 每段采用时间中点速度。
        scale = max(
            0.0,
            1.0 - (elapsed + step / 2) / stop_time,
        )
        brake_v = v * scale
        brake_omega = omega * scale

        if segment_collides(
            pose, brake_v, brake_omega, step,
            obstacles, config,
        ):
            return False

        pose = move(
            *pose, brake_v, brake_omega, step
        )
        elapsed += step

    return True


def free_path_length(pose, v, omega, obstacles, config):
    """
    估计沿恒定速度轨迹到首次碰撞的距离，
    最多搜索 distance_cap 米。

    使用小段保守检查，返回最后确认安全的距离。
    """
    if pose_collides(pose, obstacles, config):
        return 0.0

    # 圆形机器人原地旋转不改变占据区域。
    if abs(v) < 1e-8:
        return config.distance_cap

    travelled = 0.0

    while config.distance_cap - travelled > 1e-10:
        step = min(
            config.simulation_dt,
            (config.distance_cap - travelled) / abs(v),
        )

        if segment_collides(
            pose, v, omega, step, obstacles, config
        ):
            return travelled

        pose = move(*pose, v, omega, step)
        travelled += abs(v) * step

    return config.distance_cap


def heading_score(x, y, theta, goal_x, goal_y):
    if math.hypot(goal_x - x, goal_y - y) < 1e-10:
        return 1.0

    target_angle = math.atan2(goal_y - y, goal_x - x)
    error = target_angle - theta
    error = math.atan2(math.sin(error), math.cos(error))

    return 1.0 - abs(error) / math.pi


def distance_score(distance, distance_cap):
    return max(0.0, min(distance / distance_cap, 1.0))


def speed_score(v, max_v):
    # 当前版本只考虑非负前进速度
    return max(0.0, min(v / max_v, 1.0))


def total_score(heading, distance, speed, config):
    return (
        config.heading_weight * heading
        + config.distance_weight * distance
        + config.speed_weight * speed
    )


def dwa_step(state, goal, obstacles, config):
    window = calculate_dynamic_window(state, config)
    candidates = []
    best_parts = None
    best_score = float("-inf")
    best_control = None
    best_trajectory = None

    for v, omega in sample_velocities(window, config):
        if not is_admissible(
            state, v, omega, obstacles, config
        ):
            continue

        trajectory = predict_trajectory(
            state[:3],
            v,
            omega,
            config.prediction_time,
            config.simulation_dt,
        )

        # 不给预测时段内会碰撞的候选评分。
        if not trajectory_is_clear(
            state[:3], v, omega,
            config.prediction_time,
            obstacles, config,
        ):
            continue

        candidates.append(trajectory)
        heading = heading_score(
            *trajectory[-1], *goal
        )

        
        free_distance = free_path_length(
            state[:3], v, omega, obstacles, config
        )

        distance = distance_score(
            free_distance,
            config.distance_cap,
        )

# 停车与原地转向仍允许参与选择，
# 但不因为“没有向前运动”而获得最高前方距离奖励。
        if abs(v) < 1e-6:
            distance = 0.0

        speed = speed_score(v, config.max_v)
        score = total_score(
            heading, distance, speed, config
        )

        if score > best_score:
            best_parts = (heading, distance, speed)
            best_score = score
            best_control = (v, omega)
            best_trajectory = trajectory
    if best_control is not None:
        if abs(best_control[0]) < 1e-6:
            print(
            "选择低速候选：",
            best_control,
            "朝向、距离、速度分数：",
            best_parts,
            )
    return best_control, best_trajectory, candidates


@dataclass
class MovingObstacle:
    # 仿真世界持有运动参数；这些参数不传给 DWA。
    start: tuple
    end: tuple
    speed: float
    radius: float = 0.16
    phase: float = 0.0  # 初始已走距离，米

    def position(self, time):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.hypot(dx, dy)
        if length <= 0 or self.speed < 0:
            raise ValueError("往返端点必须不同，速度必须非负")
        distance = (self.phase + self.speed * time) % (2 * length)
        if distance > length:
            distance = 2 * length - distance
        return (self.start[0] + dx * distance / length,
                self.start[1] + dy * distance / length)


def observe_obstacles(static_obstacles, movers, time):
    """规划器只接收当前圆心和半径，不接收运动轨迹或速度。"""
    return list(static_obstacles) + [
        (*item.position(time), item.radius) for item in movers
    ]


def execute_world_step(state, control, time, static_obstacles,
                       movers, config):
    """世界推进与规划分离；小步保守检测双方运动中的实体碰撞。"""
    v, omega = control
    pose = state[:3]
    elapsed = 0.0
    while config.control_dt - elapsed > 1e-10:
        dt = min(0.01, config.control_dt - elapsed)
        middle = move(*pose, v, omega, dt / 2)
        for ox, oy, radius in static_obstacles:
            if math.hypot(middle[0]-ox, middle[1]-oy) <= (
                config.robot_radius + radius + abs(v)*dt/2
            ):
                return (*pose, v, omega), time+elapsed, True
        for item in movers:
            ox, oy = item.position(time + elapsed + dt/2)
            # 双方在半步内移动距离的上界，防止穿透漏检。
            bound = (abs(v) + item.speed) * dt / 2
            if math.hypot(middle[0]-ox, middle[1]-oy) <= (
                config.robot_radius + item.radius + bound
            ):
                return (*pose, v, omega), time+elapsed, True
        pose = move(*pose, v, omega, dt)
        elapsed += dt
    return (*pose, v, omega), time+elapsed, False


def main():
    config = Config()
    state = (0., 0., 0., 0., 0.)
    goal = (4., 2.)
    static_obstacles = [(1.8, 0.7, 0.3), (3., 1.8, 0.3)]
    # 修改端点和 speed 可改变红点的往返路线及速度。
    # 线段内匀速，到端点时立即反向；仅为仿真设定。
    movers = [
        MovingObstacle((1.0, 1.5), (2.5, 1.5), 0.18),
        MovingObstacle((3.8, -0.5), (3.8, 3.0), 0.22),
    ]
    time = 0.0
    steps = 0
    finished = False
    history = [state[:2]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set(xlim=(-1, 5.5), ylim=(-1.5, 4),
           xlabel='x / m', ylabel='y / m',
           title='DWA: moving obstacles, current positions only')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.plot(0, 0, 'bo', label='Start')
    ax.plot(*goal, marker='*', color='darkorange', markersize=16,
            linestyle='None', label='Goal')
    ax.add_patch(Circle(goal, config.goal_tolerance, fill=False,
                        edgecolor='darkorange', linestyle=':'))
    for x, y, radius in static_obstacles:
        ax.add_patch(Circle((x,y), radius, color='gray'))
        ax.add_patch(Circle((x,y), radius+config.robot_radius+config.safety_margin,
                            fill=False, edgecolor='gray', linestyle='--', alpha=.5))
    moving_patches = []
    for i, item in enumerate(movers):
        patch = Circle(item.position(0), item.radius, color='red',
                       label='Moving obstacle' if i == 0 else None)
        ax.add_patch(patch)
        moving_patches.append(patch)
    robot = Circle(state[:2], config.robot_radius, color='royalblue', alpha=.8,
                   label='Robot')
    ax.add_patch(robot)
    candidates_artist = LineCollection([], colors='gray', alpha=.3,
                                      linewidths=.8, label='Safe snapshot candidates')
    ax.add_collection(candidates_artist)
    selected, = ax.plot([], [], 'g--', linewidth=2.5, label='Selected prediction')
    path, = ax.plot([], [], 'b-', linewidth=2, label='Executed path')
    heading, = ax.plot([], [], 'k-', linewidth=2)
    text = ax.text(.02, .98, '', transform=ax.transAxes, va='top')
    ax.legend(loc='lower right', fontsize=8)

    def draw():
        x, y, theta = state[:3]
        robot.center = (x,y)
        heading.set_data([x,x+.38*math.cos(theta)], [y,y+.38*math.sin(theta)])
        path.set_data([p[0] for p in history], [p[1] for p in history])
        for patch, item in zip(moving_patches, movers):
            patch.center = item.position(time)

    def finish(message):
        nonlocal finished
        finished = True
        text.set_text(message)
        selected.set_data([], [])
        candidates_artist.set_segments([])
        animation.event_source.stop()

    def update(_):
        nonlocal state, time, steps
        if finished:
            return
        if math.hypot(goal[0]-state[0],goal[1]-state[1]) <= config.goal_tolerance:
            finish('Goal region reached — simulation ended')
            return
        if steps >= 600:
            finish('Maximum steps reached')
            return
        # DWA 看到的是当前时刻快照，规划期间将障碍当作静止。
        snapshot = observe_obstacles(static_obstacles, movers, time)
        control, trajectory, candidates = dwa_step(state, goal, snapshot, config)
        candidates_artist.set_segments([
            [(p[0],p[1]) for p in candidate] for candidate in candidates
        ])
        if control is None:
            finish('No admissible control — entire simulation paused')
            return
        selected.set_data([p[0] for p in trajectory], [p[1] for p in trajectory])
        state, time, collision = execute_world_step(
            state, control, time, static_obstacles, movers, config
        )
        steps += 1
        history.append(state[:2])
        draw()
        if collision:
            finish('Collision / conservative contact detected — simulation stopped')
            return
        text.set_text(
            f'Step: {steps}   Time: {time:.1f} s\n'
            f'v: {state[3]:.2f} m/s   omega: {state[4]:.2f} rad/s\n'
            f'Candidates: {len(candidates)}\n'
            'Planner observes current obstacle positions only'
        )

    draw()
    animation = FuncAnimation(fig, update, init_func=draw,
                              interval=config.control_dt*1000, blit=False,
                              cache_frame_data=False)
    plt.show()


if __name__ == '__main__':
    main()
