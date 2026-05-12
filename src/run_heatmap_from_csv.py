from visualization.heatmap_renderer import render_heatmap_from_csv


def main():
    csv_path = r"logs\company_gaze_result.csv"
    output_path = r"figures\company_gaze_heatmap.png"

    render_heatmap_from_csv(
        csv_path=csv_path,
        output_path=output_path,
        screen_width=2560,
        screen_height=1440
    )


if __name__ == "__main__":
    main()