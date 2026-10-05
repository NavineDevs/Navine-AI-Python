from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from navine.utils.paths import get_project_root

SENSOR_COUNT = 5
HIDDEN = 8
INPUTS = SENSOR_COUNT + 2
OUTPUTS = 2
POP_SIZE = 36
ELITE = 8
MUTATION = 0.12
TRACK_W, TRACK_H = 960, 640


def _backend_name() -> str:
    try:
        import jax  # noqa: F401

        return "jax"
    except Exception:
        return "numpy"


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _backend_name() == "jax":
        try:
            import jax.numpy as jnp

            return np.asarray(jnp.dot(jnp.asarray(a), jnp.asarray(b)))
        except Exception:
            pass
    return a @ b


def _tanh(x: np.ndarray) -> np.ndarray:
    if _backend_name() == "jax":
        try:
            import jax.numpy as jnp

            return np.asarray(jnp.tanh(jnp.asarray(x)))
        except Exception:
            pass
    return np.tanh(x)


def genome_size() -> int:
    return INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS


def random_genome(scale: float = 0.45) -> np.ndarray:
    return np.random.randn(genome_size()).astype(np.float32) * scale


def forward(genome: np.ndarray, sensors: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray(sensors, dtype=np.float32)
    i = 0
    w1 = genome[i : i + INPUTS * HIDDEN].reshape(INPUTS, HIDDEN)
    i += INPUTS * HIDDEN
    b1 = genome[i : i + HIDDEN]
    i += HIDDEN
    w2 = genome[i : i + HIDDEN * OUTPUTS].reshape(HIDDEN, OUTPUTS)
    i += HIDDEN * OUTPUTS
    b2 = genome[i : i + OUTPUTS]
    h = _tanh(_matmul(x, w1) + b1)
    y = _tanh(_matmul(h, w2) + b2)
    return float(y[0]), float(y[1])


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.random.rand(a.size) < 0.5
    child = np.where(mask, a, b).astype(np.float32)
    noise = np.random.randn(a.size).astype(np.float32) * MUTATION
    mutate = np.random.rand(a.size) < 0.18
    child = np.where(mutate, child + noise, child)
    return child


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float


def build_track() -> Tuple[List[Segment], List[Tuple[float, float]], Tuple[float, float, float]]:
    cx, cy = TRACK_W / 2, TRACK_H / 2
    outer_rx, outer_ry = 420, 260
    inner_rx, inner_ry = 250, 130
    walls: List[Segment] = []
    checkpoints: List[Tuple[float, float]] = []
    steps = 48
    for i in range(steps):
        t0 = (i / steps) * math.tau
        t1 = ((i + 1) / steps) * math.tau
        walls.append(
            Segment(
                cx + outer_rx * math.cos(t0),
                cy + outer_ry * math.sin(t0),
                cx + outer_rx * math.cos(t1),
                cy + outer_ry * math.sin(t1),
            )
        )
        walls.append(
            Segment(
                cx + inner_rx * math.cos(t0),
                cy + inner_ry * math.sin(t0),
                cx + inner_rx * math.cos(t1),
                cy + inner_ry * math.sin(t1),
            )
        )
        checkpoints.append((cx + 335 * math.cos(t0), cy + 195 * math.sin(t0)))
    start = (cx + 335.0, cy, -math.pi / 2)
    return walls, checkpoints, start


def _ray_hit(x: float, y: float, ang: float, walls: List[Segment], max_dist: float = 220.0) -> float:
    dx, dy = math.cos(ang), math.sin(ang)
    best = max_dist
    for wall in walls:
        hit = _seg_ray(x, y, dx, dy, wall)
        if hit is not None and 0 < hit < best:
            best = hit
    return best / max_dist


def _seg_ray(x: float, y: float, dx: float, dy: float, wall: Segment) -> Optional[float]:
    x3, y3, x4, y4 = wall.x1, wall.y1, wall.x2, wall.y2
    den = (x4 - x3) * dy - (y4 - y3) * dx
    if abs(den) < 1e-8:
        return None
    t = ((x3 - x) * dy - (y3 - y) * dx) / den
    u = ((x3 - x) * (y4 - y3) - (y3 - y) * (x4 - x3)) / den
    if 0 <= t <= 1 and u >= 0:
        return u
    return None


def _point_in_wall(x: float, y: float, walls: List[Segment], thick: float = 10.0) -> bool:
    for wall in walls:
        if _dist_point_seg(x, y, wall) < thick:
            return True
    return False


def _dist_point_seg(x: float, y: float, wall: Segment) -> float:
    vx, vy = wall.x2 - wall.x1, wall.y2 - wall.y1
    wx, wy = x - wall.x1, y - wall.y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(wx, wy)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(x - wall.x2, y - wall.y2)
    b = c1 / c2
    return math.hypot(x - (wall.x1 + b * vx), y - (wall.y1 + b * vy))


class Car:
    def __init__(self, x: float, y: float, angle: float, genome: Optional[np.ndarray] = None, color=(80, 200, 255)):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.genome = genome if genome is not None else random_genome()
        self.alive = True
        self.fitness = 0.0
        self.checkpoint = 0
        self.color = color
        self.human = False
        self.distance = 0.0

    def sensors(self, walls: List[Segment]) -> List[float]:
        spreads = (-0.9, -0.45, 0.0, 0.45, 0.9)
        vals = [_ray_hit(self.x, self.y, self.angle + a, walls) for a in spreads]
        vals.append(max(0.0, min(1.0, self.speed / 8.0)))
        vals.append((self.angle % math.tau) / math.tau)
        return vals

    def step(self, walls: List[Segment], checkpoints: List[Tuple[float, float]], steer: float, throttle: float) -> None:
        if not self.alive:
            return
        self.angle += steer * 0.085
        self.speed += throttle * 0.22
        self.speed *= 0.985
        self.speed = max(-2.0, min(7.5, self.speed))
        nx = self.x + math.cos(self.angle) * self.speed
        ny = self.y + math.sin(self.angle) * self.speed
        if _point_in_wall(nx, ny, walls):
            self.alive = False
            self.speed = 0.0
            return
        self.distance += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        target = checkpoints[self.checkpoint % len(checkpoints)]
        if math.hypot(self.x - target[0], self.y - target[1]) < 42:
            self.checkpoint += 1
        self.fitness = self.checkpoint * 120.0 + self.distance * 0.35


def scripted_policy(sensors: Sequence[float]) -> Tuple[float, float]:
    left, mid_l, mid, mid_r, right, speed, _ang = sensors
    steer = (right + mid_r) - (left + mid_l)
    steer = max(-1.0, min(1.0, steer * 1.4))
    throttle = 0.7 if mid > 0.35 else -0.4
    if speed > 0.85:
        throttle *= 0.5
    return steer, throttle


def save_best(genome: np.ndarray) -> Path:
    path = get_project_root() / "checkpoints" / "games_racing"
    path.mkdir(parents=True, exist_ok=True)
    out = path / "best_genome.npy"
    np.save(out, genome)
    meta = {
        "backend": _backend_name(),
        "inputs": INPUTS,
        "hidden": HIDDEN,
        "outputs": OUTPUTS,
        "style": "self-driving genetic pygame race",
        "inspired_by": "GGWHjAyKJCA",
    }
    (path / "best_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_best() -> Optional[np.ndarray]:
    path = get_project_root() / "checkpoints" / "games_racing" / "best_genome.npy"
    if not path.exists():
        return None
    return np.load(path)


def run_game(mode: str = "evolve", generations: int = 40, headless_steps: int = 0) -> dict:
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required. Run: pip install pygame") from exc

    walls, checkpoints, start = build_track()
    pygame.init()
    screen = pygame.display.set_mode((TRACK_W, TRACK_H))
    pygame.display.set_caption(f"Navine AI - Python Racing ({_backend_name()})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    population = [random_genome() for _ in range(POP_SIZE)]
    best = load_best()
    if best is not None and best.size == genome_size():
        population[0] = best
    generation = 0
    cars: List[Car] = []
    human: Optional[Car] = None
    frame = 0
    finished = False
    report = {"backend": _backend_name(), "mode": mode, "generations": 0, "best_fitness": 0.0}

    def reset_gen(genomes: List[np.ndarray]) -> List[Car]:
        out = []
        for i, g in enumerate(genomes):
            color = (80, 220, 255) if i else (255, 210, 70)
            out.append(Car(start[0], start[1], start[2], genome=g, color=color))
        return out

    def reset_race() -> List[Car]:
        g = load_best()
        if g is None:
            g = random_genome()
        ai = Car(start[0], start[1] - 18, start[2], genome=g, color=(80, 220, 255))
        bot = Car(start[0], start[1] + 18, start[2], genome=None, color=(255, 120, 120))
        bot.genome = None
        return [ai, bot]

    if mode == "evolve":
        cars = reset_gen(population)
    elif mode == "race":
        cars = reset_race()
    else:
        g = load_best() or random_genome()
        human = Car(start[0], start[1] - 18, start[2], genome=g, color=(120, 255, 140))
        human.human = True
        ai = Car(start[0], start[1] + 18, start[2], genome=g.copy(), color=(80, 220, 255))
        cars = [human, ai]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        for car in cars:
            if not car.alive:
                continue
            if car.human:
                steer = 0.0
                throttle = 0.0
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    steer = -1.0
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    steer = 1.0
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    throttle = 1.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    throttle = -0.8
            elif car.genome is None:
                steer, throttle = scripted_policy(car.sensors(walls))
            else:
                steer, throttle = forward(car.genome, car.sensors(walls))
            car.step(walls, checkpoints, steer, throttle)

        frame += 1
        alive = sum(1 for c in cars if c.alive)
        if mode == "evolve" and (alive == 0 or frame > 900):
            ranked = sorted(cars, key=lambda c: c.fitness, reverse=True)
            best_fit = ranked[0].fitness
            report["best_fitness"] = max(report["best_fitness"], best_fit)
            save_best(ranked[0].genome)
            elites = [c.genome.copy() for c in ranked[:ELITE]]
            population = elites[:]
            while len(population) < POP_SIZE:
                a, b = random.sample(elites, 2)
                population.append(crossover(a, b))
            generation += 1
            report["generations"] = generation
            cars = reset_gen(population)
            frame = 0
            if generation >= generations:
                finished = True
                running = False

        screen.fill((12, 16, 28))
        for wall in walls:
            pygame.draw.line(screen, (70, 90, 140), (wall.x1, wall.y1), (wall.x2, wall.y2), 3)
        for i, (cx, cy) in enumerate(checkpoints[::4]):
            pygame.draw.circle(screen, (40, 60, 40), (int(cx), int(cy)), 4)
        for car in cars:
            if not car.alive:
                continue
            pygame.draw.circle(screen, car.color, (int(car.x), int(car.y)), 7)
            tip = (car.x + math.cos(car.angle) * 14, car.y + math.sin(car.angle) * 14)
            pygame.draw.line(screen, (255, 255, 255), (car.x, car.y), tip, 2)

        best_now = max((c.fitness for c in cars), default=0.0)
        hud = [
            f"mode={mode} backend={_backend_name()}",
            f"gen={generation} alive={alive} best={best_now:.1f}",
            "ESC quit | evolve watches AI learn via genetic training",
            "Arrows/WASD if play mode",
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (220, 230, 255)), (12, 10 + i * 20))
        pygame.display.flip()
        clock.tick(60)
        if headless_steps and frame >= headless_steps:
            running = False

    pygame.quit()
    report["finished"] = finished
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv or sys.argv[1:])
    mode = "evolve"
    generations = 40
    if args:
        mode = args[0].strip().lower()
    if len(args) > 1 and args[1].isdigit():
        generations = int(args[1])
    if mode not in ("evolve", "race", "play"):
        mode = "evolve"
    report = run_game(mode=mode, generations=generations)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
