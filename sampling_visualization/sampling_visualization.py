import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

# 设置中文字体，自动适配常见环境
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 通用采样器 ---
def general_sampling(points, k, mode="arithmetic"):
    n = len(points)
    if k >= n:
        return list(range(n))
    selected = [0]
    if k > 1:
        dists = cdist(points, points[selected])[:, 0]
        second = np.argmax(dists)
        if second != 0:
            selected.append(second)
    remaining = set(range(n)) - set(selected)
    while len(selected) < k and remaining:
        current_selected_points = points[selected]
        candidate_list = list(remaining)
        candidate_points = points[candidate_list]
        dist_to_selected = cdist(candidate_points, current_selected_points)
        min_distance = 1e-12
        dist_to_selected = np.maximum(dist_to_selected, min_distance)
        if mode == "arithmetic":
            agg = np.sum(dist_to_selected, axis=1)
            best_idx = np.argmax(agg)
        elif mode == "geometric":
            agg = np.sum(np.log(dist_to_selected), axis=1)
            best_idx = np.argmax(agg)
        elif mode == "harmonic":
            agg = np.sum(1.0 / dist_to_selected, axis=1)
            best_idx = np.argmin(agg)
        elif mode == "power2":
            agg = np.sum(dist_to_selected ** 2, axis=1)
            best_idx = np.argmax(agg)
        else:
            raise ValueError("Unknown mode")
        best_candidate = candidate_list[best_idx]
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return selected

# --- 数据生成 ---
def generate_2d(n):
    theta = np.random.rand(n) * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    return np.stack([x, y], axis=1)

def generate_3d(n):
    phi = np.random.rand(n) * 2 * np.pi
    costheta = np.random.rand(n) * 2 - 1
    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)

# --- 可视化 ---
def plot_sampling(points, selected_idx, title, ax):
    dim = points.shape[1]
    if dim == 2:
        ax.scatter(points[:, 0], points[:, 1], c="#cccccc", s=10, alpha=0.1)
        ax.scatter(points[selected_idx, 0], points[selected_idx, 1], c="red", s=20, edgecolor="k")
        ax.set_aspect("equal")
    elif dim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = plt.subplot(ax.get_subplotspec(), projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#cccccc", s=10, alpha=0.1)
        ax.scatter(points[selected_idx, 0], points[selected_idx, 1], points[selected_idx, 2], c="red", s=20, edgecolor="k")
        ax.set_box_aspect([1,1,1])
    ax.set_title(title)
    ax.legend()

if __name__ == "__main__":
    n = 10
    sample_ratio = 0.2
    k2 = max(2, int(2*3*n * sample_ratio))
    k3 = max(2, int(4*3*n**2 * sample_ratio))
    # 2D
    pts2d = generate_2d(int(2*3*n))
    # 3D
    pts3d = generate_3d(int(4*3*n**2))
    
    modes = ["arithmetic", "geometric", "harmonic", "power2"]
    mode_names = ["算术平均", "几何平均", "调和平均", "平方平均"]
    
    fig, axes = plt.subplots(2, len(modes), figsize=(16, 9))
    for i, (mode, mode_name) in enumerate(zip(modes, mode_names)):
        idx2 = general_sampling(pts2d, k2, mode)
        idx3 = general_sampling(pts3d, k3, mode)
        plot_sampling(pts2d, idx2, f"2D-{mode_name}", axes[0, i])
        plot_sampling(pts3d, idx3, f"3D-{mode_name}", axes[1, i])
    plt.tight_layout()
    outdir = os.path.dirname(__file__)
    outimg = os.path.join(outdir, "all_sampling_methods.png")
    plt.savefig(outimg, dpi=600)
    plt.show()
