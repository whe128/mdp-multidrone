import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt
from multi_drone_config import MultiDroneConfig
from typing import Optional, Tuple, Dict
from vedo import Plotter, Box, Sphere, Cylinder, Line, Axes

def _make_action_vectors(change_altitude: bool) -> np.ndarray:
    if change_altitude:
        # 26 directions in 3D (no zero move)
        return np.array(
            [[dx, dy, dz]
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             for dz in (-1, 0, 1)
             if not (dx == 0 and dy == 0 and dz == 0)],
            dtype=np.int32
        )
    else:
        # 8 directions in XY plane (dz=0, no zero move)
        return np.array(
            [[dx, dy, 0]
             for dx in (-1, 0, 1)
             for dy in (-1, 0, 1)
             if not (dx == 0 and dy == 0)],
            dtype=np.int32
        )

class MultiDroneUnc:    
    def __init__(self, path: str):
        self.cfg = self._load_config(path)

        assert self.cfg.start_positions is not None, "start_positions can't be empty"
        self.N = len(self.cfg.start_positions)       
        self.rng = np.random.default_rng(self.cfg.seed)

        self.positions = np.zeros((self.N, 3), dtype=np.int32)
        self.reached = np.zeros((self.N,), dtype=bool)
        self.t = 0

        self.obstacles = np.zeros(self.cfg.grid_size, dtype=bool)
        self.dist_field = np.zeros(self.cfg.grid_size, dtype=np.float32)

        # vedo visualization members
        self._plotter = None
        self._uuv_visuals = []
        self._traj_lines = []
        self._obstacle_meshes = []
        self._goal_mesh = None
        self._action_vectors = _make_action_vectors(self.cfg.change_altitude)
    
    def get_config(self):
        return self.cfg

    @property
    def num_actions(self) -> int:        
        return (self._action_vectors.shape[0]) ** self.N

    # ---------- Core API ----------
    def reset(self) -> np.ndarray:        
        X, Y, Z = self.cfg.grid_size

        # Obstacles
        self.obstacles.fill(False)

        if self.cfg.obstacle_cells is not None:
            for (x, y, z) in self.cfg.obstacle_cells:
                # Check bounds
                assert 0 <= x < X and 0 <= y < Y and 0 <= z < Z, \
                    f"Obstacle {(x, y, z)} is outside environment bounds {self.cfg.grid_size}"

                # Check not on goal plane
                assert x != X - 1, \
                    f"Obstacle {(x, y, z)} is placed on the goal plane (x = {X-1}), which is not allowed."

                self.obstacles[x, y, z] = True


        # distance field
        free = (~self.obstacles).astype(np.uint8)
        self.dist_field = distance_transform_edt(free).astype(np.float32)

        # Start positions
        self.positions = np.array(self.cfg.start_positions, dtype=np.int32)

        # Goal positions
        assert self.cfg.goal_positions is not None, "goal_positions must be provided"
        self.goals = np.array(self.cfg.goal_positions, dtype=np.int32)
        assert self.goals.shape == self.positions.shape, \
            "goal_positions must match number of drones"

        # Reached flags
        self.reached = np.all(self.positions == self.goals, axis=1)

        # reset timestep
        self.t = 0

        # build unified state
        state = np.zeros((self.N, 4), dtype=np.int32)
        state[:, :3] = self.positions
        state[:, 3] = self.reached.astype(np.int32)

        self._init_plot()
        return state

    def step(self, action_int: int) -> Tuple[np.ndarray, float, bool, Dict]:
        state = np.hstack([self.positions, self.reached[:,None].astype(np.int32)])
        next_state, reward, done, info = self._transition_dynamics(state, action_int)

        # update the "real" env state
        self.positions = next_state[:, :3]
        self.reached   = next_state[:, 3].astype(bool)
        self.t += 1

        # enforce max_num_steps
        if self.t >= self.cfg.max_num_steps:
            done = True
            info["max_steps_reached"] = True

        self._update_plot()
        return next_state.copy(), reward, done, info

    def simulate(self,
                 state: np.ndarray,
                 action: int) -> Tuple[np.ndarray, float, bool, Dict]:        
        next_state, reward, done, info = self._transition_dynamics(state, action)
        return next_state, reward, done, info

    def _transition_dynamics(self,
                             state: np.ndarray,
                             action_int: int,
                            ) -> Tuple[np.ndarray, float, bool, Dict]:
        X, Y, Z = self.cfg.grid_size
        prev_pos = state[:, :3].astype(np.int32)
        reached = state[:, 3].astype(bool)
        actions = self._decode_action(action_int)

        # success probabilities (vectorized)
        d = self.dist_field[prev_pos[:, 0], prev_pos[:, 1], prev_pos[:, 2]]
        p_succ = d / (d + self.cfg.alpha)
        p_succ[reached] = 0.0   # reached drones don't move

        num_actions = self._action_vectors.shape[0]

        # Precompute all neighbors for all vehicles (N, num_actions, 3)
        neighs = prev_pos[:, None, :] + self._action_vectors[None, :, :]

        # valid moves (N, num_actions)
        valid = (
            (neighs[..., 0] >= 0) & (neighs[..., 0] < X) &
            (neighs[..., 1] >= 0) & (neighs[..., 1] < Y) &
            (neighs[..., 2] >= 0) & (neighs[..., 2] < Z)
        )

        # build probabilities for each vehicle (N, num_actions)
        probs = np.full((self.N, num_actions),
                        (1 - p_succ)[:, None] / (num_actions - 1),
                        dtype=np.float32)
        probs[np.arange(self.N), actions] = p_succ
        probs[~valid] = 0.0

        # normalize per vehicle
        probs /= probs.sum(axis=1, keepdims=True)

        # sample moves in one go
        choices = np.array([self.rng.choice(num_actions, p=probs[i])
                            for i in range(self.N)], dtype=np.int32)
        targets = neighs[np.arange(self.N), choices]

        # --- lock reached drones at their goal positions ---
        targets[reached] = prev_pos[reached]

        # obstacle collisions
        obstacles_hit = self.obstacles[targets[:, 0], targets[:, 1], targets[:, 2]]
        collisions = int(obstacles_hit.sum())

        # apply moves (stay put if hit obstacle)
        next_pos = np.where(obstacles_hit[:, None], prev_pos, targets)

        # just reached / reached
        just_reached = (~reached) & np.all(next_pos == self.goals, axis=1)
        next_reached = reached | just_reached

        # vehicle–vehicle collisions (still include reached drones as solid)
        _, counts = np.unique(next_pos, axis=0, return_counts=True)
        veh_collisions = int(counts[counts > 1].sum())

        # rewards
        reward = self.cfg.step_cost * self.N
        reward += self.cfg.collision_penalty * (collisions + veh_collisions)
        reward += self.cfg.goal_reward * np.sum(just_reached)

        # combine into next_state
        next_state = np.zeros((self.N, 4), dtype=np.int32)
        next_state[:, :3] = next_pos
        next_state[:, 3] = next_reached.astype(np.int32)

        # terminal if: all reached, obstacle collision, or vehicle collision
        done = next_reached.all() or (collisions > 0) or (veh_collisions > 0)

        info = dict(
            num_collisions=collisions,
            num_vehicle_collisions=veh_collisions,
            success=next_reached.all(),
        )
        return next_state, reward, done, info


    def _load_config(self, path: str) -> MultiDroneConfig:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        if "grid_size" in data and isinstance(data["grid_size"], list):
            data["grid_size"] = tuple(data["grid_size"])

        return MultiDroneConfig(**data)

    # ---------- Action encoding ----------
    def _decode_action(self, action_int: int) -> np.ndarray:        
        num_per_vehicle = self._action_vectors.shape[0]
        actions = np.zeros(self.N, dtype=np.int32)
        x = action_int
        for i in range(self.N):
            actions[i] = x % num_per_vehicle
            x //= num_per_vehicle
        return actions

    def _encode_action(self, actions: np.ndarray) -> int:        
        num_per_vehicle = self._action_vectors.shape[0]
        base = 1
        out = 0
        for a in actions:
            out += int(a) * base
            base *= num_per_vehicle
        return out

    # ---------- Visualization ----------
    def _grid_to_world(self, pos: np.ndarray, vs: float = 1.0) -> np.ndarray:
        return pos * vs

    def _init_plot(self):
        self._plotter = Plotter(interactive=False)
        self._uuv_visuals = []
        self._traj_lines = []
        self._obstacle_meshes = []
        self._goal_meshes = []

        X, Y, Z = self.cfg.grid_size
        vs = 1.0        

        # obstacles (center each cube on the cell center)
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    if self.obstacles[x, y, z]:
                        center = self._grid_to_world(np.array([x, y, z]), vs)
                        cube = Box(pos=center.tolist(), length=vs, width=vs, height=vs).c("gray").alpha(0.5)
                        self._obstacle_meshes.append(cube)

        # goals: green cubes centered on cell centers
        for g in self.goals:   # shape (N,3)
            center = self._grid_to_world(g, vs)
            goal_box = Box(pos=center.tolist(), length=vs, width=vs, height=vs).c("green").alpha(0.6)
            self._goal_meshes.append(goal_box)

        domain = Box(
            pos=[(X*vs)/2, (Y*vs)/2, (Z*vs)/2],
            length=X*vs, width=Y*vs, height=Z*vs,
        ).wireframe().c("gray3").alpha(0.3)

        # drones
        for i in range(self.N):
            pos = self._grid_to_world(self.positions[i], vs)
            body = Sphere(pos=pos.tolist(), r=vs*0.3).c("cyan")
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*vs,
                                 pos + np.array([ 0.5, 0, 0])*vs], r=vs*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*vs,
                                 pos + np.array([0,  0.5, 0])*vs], r=vs*0.05).c("black")
            traj = Line([pos]).c("blue").lw(2)
            self._uuv_visuals.append((body, arm1, arm2))
            self._traj_lines.append(traj)

        # Create a dummy box actor that defines the bounds
        bounds = [0, X*vs, 0, Y*vs, 0, Z*vs]
        axes = dict(
            xrange=(bounds[0], bounds[1]),
            yrange=(bounds[2], bounds[3]),
            zrange=(bounds[4], bounds[5]),
            xygrid=True,
        )

        self._plotter.show(
            *self._obstacle_meshes, *self._goal_meshes,
            *[v for triple in self._uuv_visuals for v in triple],
            *self._traj_lines,
            axes=axes,
            viewup="z", interactive=False,
        )


    def _update_plot(self):
        vs = 1.0
        for i in range(self.N):
            pos = self._grid_to_world(self.positions[i], vs)
            body, arm1, arm2 = self._uuv_visuals[i]

            # update visuals
            body.pos(pos)
            self._plotter.remove(arm1)
            self._plotter.remove(arm2)
            arm1 = Cylinder(pos=[pos + np.array([-0.5, 0, 0])*vs,
                                 pos + np.array([ 0.5, 0, 0])*vs], r=vs*0.05).c("black")
            arm2 = Cylinder(pos=[pos + np.array([0, -0.5, 0])*vs,
                                 pos + np.array([0,  0.5, 0])*vs], r=vs*0.05).c("black")
            self._uuv_visuals[i] = (body, arm1, arm2)

            # update trajectory: always keep as Line
            old_pts = self._traj_lines[i].points
            old_pts = old_pts() if callable(old_pts) else old_pts

            new_pts = np.vstack([old_pts, pos])
            self._plotter.remove(self._traj_lines[i])
            self._traj_lines[i] = Line(new_pts).c("blue").lw(2)
            self._plotter.add(self._traj_lines[i])

            self._plotter.add(arm1)
            self._plotter.add(arm2)

        self._plotter.render()

    def show(self):
        """Enter interactive mode to explore scene."""
        self._plotter.interactive()