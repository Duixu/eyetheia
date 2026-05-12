import os
import glob
import pickle
import numpy as np
import cv2
import pandas as pd


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


def find_latest_pkl(folder):
    """
    在指定文件夹中寻找最新的 pkl 文件。
    主要用于兼容 EyeTheia 原有输出。
    """
    pkl_files = glob.glob(os.path.join(folder, "*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in: {folder}")

    latest_file = max(pkl_files, key=os.path.getmtime)
    return latest_file


def load_points_from_pkl(pkl_path):
    """
    从 pkl 文件中读取 gaze 点。
    要求 pkl 中的数据格式类似：
    [(x1, y1), (x2, y2), ...]
    """
    with open(pkl_path, "rb") as f:
        points = pickle.load(f)

    return points


def load_points_from_csv(csv_path):
    """
    从标准 CSV 中读取 gaze 点。
    CSV 必须包含 gaze_x 和 gaze_y 两列。
    """
    df = pd.read_csv(csv_path)

    if "gaze_x" not in df.columns or "gaze_y" not in df.columns:
        raise ValueError("CSV 文件中必须包含 gaze_x 和 gaze_y 两列")

    points = []

    for _, row in df.iterrows():
        x = row["gaze_x"]
        y = row["gaze_y"]
        points.append((x, y))

    return points


def render_heatmap_from_points(
    points,
    output_path,
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT,
    sigma=35,
    background_alpha=0.55,
    heatmap_alpha=0.45
):
    """
    根据 gaze 点绘制热力图。
    points: [(x, y), (x, y), ...]
    """

    if points is None or len(points) == 0:
        print("[WARNING] No gaze data found.")
        return

    heatmap = np.zeros((screen_height, screen_width), dtype=np.float32)

    valid_count = 0

    for point in points:
        x, y = point
        x = int(x)
        y = int(y)

        if 0 <= x < screen_width and 0 <= y < screen_height:
            heatmap[y, x] += 1
            valid_count += 1

    if valid_count == 0:
        raise ValueError(
            "没有有效 gaze 点。请检查 gaze_x / gaze_y 是否在屏幕坐标范围内。"
        )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    heatmap_norm = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    background = np.ones(
        (screen_height, screen_width, 3),
        dtype=np.uint8
    ) * 255

    result = cv2.addWeighted(
        background,
        background_alpha,
        heatmap_color,
        heatmap_alpha,
        0
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)

    print(f"[INFO] valid gaze points: {valid_count}")
    print(f"[INFO] heatmap saved to: {output_path}")


def render_heatmap_from_pkl(
    pkl_path,
    output_path,
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT
):
    """
    从 pkl 文件生成热力图。
    """
    points = load_points_from_pkl(pkl_path)

    render_heatmap_from_points(
        points=points,
        output_path=output_path,
        screen_width=screen_width,
        screen_height=screen_height
    )


def render_heatmap_from_csv(
    csv_path,
    output_path,
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT
):
    """
    从 CSV 文件生成热力图。
    """
    points = load_points_from_csv(csv_path)

    render_heatmap_from_points(
        points=points,
        output_path=output_path,
        screen_width=screen_width,
        screen_height=screen_height
    )


def render_latest_pkl_heatmap(
    experiment_dir,
    output_path,
    screen_width=SCREEN_WIDTH,
    screen_height=SCREEN_HEIGHT
):
    """
    兼容你原来的逻辑：
    自动寻找 experiment_dir 下面最新的 pkl 文件，然后生成热力图。
    """
    pkl_path = find_latest_pkl(experiment_dir)
    print(f"[INFO] using gaze data file: {pkl_path}")

    render_heatmap_from_pkl(
        pkl_path=pkl_path,
        output_path=output_path,
        screen_width=screen_width,
        screen_height=screen_height
    )

def generate_heatmap_image(
    points,
    screen_width=2560,
    screen_height=1440,
    sigma=35
):
    """
    根据 gaze 点生成热力图图像，并返回图像矩阵。
    不直接保存，只返回 result。
    """
    if points is None or len(points) == 0:
        return None

    heatmap = np.zeros((screen_height, screen_width), dtype=np.float32)

    valid_count = 0
    for x, y in points:
        x = int(x)
        y = int(y)
        if 0 <= x < screen_width and 0 <= y < screen_height:
            heatmap[y, x] += 1
            valid_count += 1

    if valid_count == 0:
        return None

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

    heatmap_norm = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    background = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    result = cv2.addWeighted(background, 0.55, heatmap_color, 0.45, 0)

    return result


def save_heatmap_image(image, output_path):
    if image is None:
        print("[WARNING] No heatmap image to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[INFO] Heatmap saved to: {output_path}")