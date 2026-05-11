from visualization.heatmap_renderer import render_latest_pkl_heatmap


def main():
    experiment_dir = r"src\experiments"
    output_path = r"figures\eyetheia_gaze_heatmap.png"

    render_latest_pkl_heatmap(
        experiment_dir=experiment_dir,
        output_path=output_path,
        screen_width=2560,
        screen_height=1440
    )


if __name__ == "__main__":
    main()