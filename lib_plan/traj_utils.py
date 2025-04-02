import numpy as np


def get_trajectory_straight_line(
    state_init: np.ndarray, state_fin: np.ndarray, step_size: float = 0.1
) -> np.ndarray:
    # step-size added for EGO-planner only
    start = state_init[:2].reshape(1, 2)
    end = state_fin[:2].reshape(1, 2)
    dist = np.linalg.norm(start - end, ord=2)
    num_steps = int(np.ceil(dist / step_size))

    inter = np.linspace(0, 1, num_steps, dtype=np.float32).reshape(-1, 1)
    inter = np.repeat(inter, 2, axis=-1)
    traj = start * (1 - inter) + end * inter
    return traj


def traj_extender(traj: np.ndarray, dist_thresh: float = 0.15) -> np.ndarray:
    """
    if traj points have a lot of gap in-between, pure-pursuit controller
    will have issues choosing which point to pursue.
    Hence, add intermediate points to traj
    """
    _cats = []
    for idx in range(traj.shape[0] - 1):
        _from = idx
        _to = idx + 1

        _dist = np.linalg.norm(traj[_from] - traj[_to])
        if _dist > dist_thresh:
            steps = int(np.ceil(_dist / 0.1))
            _slider = np.linspace(0, 1, steps).reshape(-1, 1)
            _ext = (
                traj[_from].reshape(1, 2) * (1 - _slider)
                + traj[_to].reshape(1, 2) * _slider
            )
            _cats.append(_ext[0:-1])
        else:
            _cats.append(traj[_from].reshape(1, 2))
        # goal pos (final point)
    _cats.append(traj[-1].reshape(1, 2))
    traj = np.concatenate(_cats, axis=0)
    return traj
