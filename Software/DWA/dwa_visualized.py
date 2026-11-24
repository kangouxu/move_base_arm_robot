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
    max_v: float = 1.8
    max_omega: float = 3.0

    # 加速度限制：本例假设加速、减速对称
    max_accel: float = 2.5
    max_angular_accel: float = 3.0

    # 制动减速度大小，必须为正
    brake_accel: float = 2.5
    brake_angular_accel: float = 3.0

    # 三个不同的时间参数
    control_dt: float = 0.1      # 多久重新选择一次速度
    prediction_time: float = 1.0
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
    candidate_trajectories = []
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

        candidate_trajectories.append(trajectory)

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
        # 但不因为没有向前运动而获得最高前方距离奖励。
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
    return best_control, best_trajectory, candidate_trajectories


def main():
    config = Config()

    state = (0.0, 0.0, 0.0, 0.0, 0.0)
    goal = (4.0, 2.0)

    # 障碍物：中心 x、中心 y、半径
    obstacles = [
        (1.8, 0.7, 0.3),
        (3.0, 1.8, 0.3),
    ]

    history_x = [state[0]]
    history_y = [state[1]]

    finished = False
    step_count = 0

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.set_aspect("equal")
    ax.set_xlim(-1, 5.5)
    ax.set_ylim(-1.5, 4)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("DWA navigation")
    ax.grid(True, alpha=0.3)

    # 起点与目标
    ax.plot(state[0], state[1], "bo", label="Start")
    ax.plot(
        goal[0], goal[1],
        marker="*", color="red", markersize=16,
        linestyle="None", label="Goal",
    )

    # 目标容差范围
    ax.add_patch(
        Circle(
            goal,
            config.goal_tolerance,
            fill=False,
            edgecolor="red",
            linestyle=":",
        )
    )

    # 障碍实体，以及考虑机器人尺寸后的禁入区域。
    for ox, oy, radius in obstacles:
        ax.add_patch(
            Circle(
                (ox, oy), radius,
                color="gray", alpha=0.9,
            )
        )

        ax.add_patch(
            Circle(
                (ox, oy),
                radius
                + config.robot_radius
                + config.safety_margin,
                fill=False,
                edgecolor="gray",
                linestyle="--",
                alpha=0.6,
            )
        )

    # 机器人实体
    robot = Circle(
        state[:2],
        config.robot_radius,
        facecolor="royalblue",
        edgecolor="navy",
        alpha=0.8,
        label="Robot",
    )
    ax.add_patch(robot)

    # 历史路径
    history_line, = ax.plot(
        [], [], "b-", linewidth=2,
        label="Executed path",
    )

    # 所有通过安全检查的候选轨迹
    candidate_lines = LineCollection(
        [], colors="gray", linewidths=0.8, alpha=0.3,
        zorder=1, label="Safe candidates",
    )
    ax.add_collection(candidate_lines)

    # 当前选中的预测轨迹
    prediction_line, = ax.plot(
        [], [], "g--", linewidth=2.5, zorder=3,
        label="Selected prediction",
    )

    # 用线段表示朝向
    heading_line, = ax.plot(
        [], [], color="black", linewidth=2,
    )

    status_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
    )

    ax.legend(loc="lower right")

    def draw_robot():
        x, y, theta = state[:3]

        robot.center = (x, y)

        heading_length = config.robot_radius * 1.5

        heading_line.set_data(
            [x, x + heading_length * math.cos(theta)],
            [y, y + heading_length * math.sin(theta)],
        )

        history_line.set_data(history_x, history_y)

    def finish(message):
        nonlocal finished

        finished = True
        candidate_lines.set_segments([])
        prediction_line.set_data([], [])
        status_text.set_text(message)

        # 停止动画计时，保留窗口。
        animation.event_source.stop()

    def update(_frame):
        nonlocal state, step_count

        if finished:
            return

        distance_to_goal = math.hypot(
            goal[0] - state[0],
            goal[1] - state[1],
        )

        if distance_to_goal <= config.goal_tolerance:
            finish("Goal region reached — simulation ended")
            return

        if step_count >= 500:
            finish("Maximum simulation steps reached")
            return

        control, trajectory, candidates = dwa_step(
            state, goal, obstacles, config
        )

        candidate_lines.set_segments([
            [(x, y) for x, y, theta in path]
            for path in candidates
        ])

        if control is None:
            finish("No admissible control — simulation paused")
            return

        v, omega = control

        # 这条预测线以执行前的位姿为起点。
        prediction_line.set_data(
            [pose[0] for pose in trajectory],
            [pose[1] for pose in trajectory],
        )

        # 实际只执行一个控制周期。
        x, y, theta = move(
            *state[:3],
            v,
            omega,
            config.control_dt,
        )

        state = (x, y, theta, v, omega)

        history_x.append(x)
        history_y.append(y)
        step_count += 1

        draw_robot()

        status_text.set_text(
            f"Step: {step_count}\n"
            f"Simulation time: "
            f"{step_count * config.control_dt:.1f} s\n"
            f"v: {v:.2f} m/s\n"
            f"omega: {omega:.2f} rad/s\n"
            f"Safe candidates: {len(candidates)}"
        )

    draw_robot()

    # 保留 animation 变量，避免动画对象被提前回收。
    animation = FuncAnimation(
        fig,
        update,
        init_func=draw_robot,
        interval=config.control_dt * 1000,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()