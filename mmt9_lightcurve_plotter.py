import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


def plot_light_curve(file_path):
    # Tkinter returns the selected filename as a STRING.
    # Convert it to a Path object so .name works correctly.
    from pathlib import Path
    csv_path = Path(file_path)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Check required columns
    if "time_seconds" not in df.columns:
        raise ValueError(
            "CSV does not contain 'time_seconds' column."
        )

    if "magnitude" not in df.columns:
        raise ValueError(
            "CSV does not contain 'magnitude' column."
        )

    # Keep only valid numerical rows
    data = df[
        ["time_seconds", "magnitude"]
    ].copy()

    data["time_seconds"] = pd.to_numeric(
        data["time_seconds"],
        errors="coerce"
    )

    data["magnitude"] = pd.to_numeric(
        data["magnitude"],
        errors="coerce"
    )

    data = data.dropna()

    if data.empty:
        raise ValueError(
            "No valid time/magnitude data found in the CSV."
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        data["time_seconds"].to_numpy(),
        data["magnitude"].to_numpy(),
        linewidth=1,
        label="Light curve"
    )

    # Plot smoothed curve if present
    if "magnitude_smooth" in df.columns:

        smooth_data = df[
            ["time_seconds", "magnitude_smooth"]
        ].copy()

        smooth_data["time_seconds"] = pd.to_numeric(
            smooth_data["time_seconds"],
            errors="coerce"
        )

        smooth_data["magnitude_smooth"] = pd.to_numeric(
            smooth_data["magnitude_smooth"],
            errors="coerce"
        )

        smooth_data = smooth_data.dropna()

        if not smooth_data.empty:
            plt.plot(
                smooth_data["time_seconds"].to_numpy(),
                smooth_data["magnitude_smooth"].to_numpy(),
                linewidth=2,
                label="Smoothed"
            )

    # Astronomical convention:
    # smaller magnitude = brighter
    plt.gca().invert_yaxis()

    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude")

    plt.title(
        f"MMT-9 Light Curve — {csv_path.name}"
    )

    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_csv():

    file_path = filedialog.askopenfilename(
        title="Select Light Curve CSV",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    file_label.config(
        text=file_path
    )

    try:
        plot_light_curve(file_path)

    except Exception as error:
        messagebox.showerror(
            "Error",
            str(error)
        )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "MMT-9 Light Curve Plotter"
)

root.geometry(
    "700x300"
)

tk.Label(
    root,
    text="MMT-9 Light Curve Plotter",
    font=("Arial", 20, "bold")
).pack(
    pady=(30, 10)
)

tk.Label(
    root,
    text=(
        "Select a CSV file containing "
        "time_seconds and magnitude."
    )
).pack(
    pady=5
)

tk.Button(
    root,
    text="SELECT CSV & PLOT",
    width=25,
    height=2,
    command=select_csv
).pack(
    pady=20
)

file_label = tk.Label(
    root,
    text="No file selected",
    wraplength=650
)

file_label.pack(
    padx=20
)

root.mainloop()
