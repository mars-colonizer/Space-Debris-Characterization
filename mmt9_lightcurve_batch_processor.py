import re
import threading
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

RUN_STAGE = 2

# Current calibration is the calibration used for the
# MMT-9 F (Folded Light Curve) plots.
# We therefore batch-process *_F.png files by default.
TARGET_PLOT_TYPE = "F"

MAX_INTERPOLATION_GAP = 3
SMOOTH_WINDOW = 7

# Output files are placed inside each track folder:
#
# TRACKID_F.png
# TRACKID_F.csv
# TRACKID_F_processed.png


# ============================================================
# MMT-9 FOLDER STRUCTURE
# ============================================================
#
# root/
# ├── 60424/
# │   ├── 29185867/
# │   │   └── 29185867_F.png
# │   └── 29641645/
# │       └── 29641645_F.png
# │
# └── 17360/
#     └── ...
#
# The script searches recursively, so the exact root path
# can be selected by the user.
#
# ============================================================


# ============================================================
# GLOBAL STOP CONTROL
# ============================================================

stop_event = threading.Event()


# ============================================================
# STAGE 1 — DIGITIZE PNG
# ============================================================

def digitize_png(input_image):

    img = Image.open(input_image).convert("RGB")
    arr = np.array(img)

    # --------------------------------------------------------
    # MMT-9 plot area
    # --------------------------------------------------------

    x_left = 56
    x_right = 865

    y_top = 33
    y_bottom = 393

    # --------------------------------------------------------
    # X-axis calibration
    # --------------------------------------------------------

    x_pixel_ticks = np.array([
        185,
        325,
        465,
        606,
        746
    ], dtype=float)

    x_value_ticks = np.array([
        -100,
        0,
        100,
        200,
        300
    ], dtype=float)

    # --------------------------------------------------------
    # Y-axis calibration
    # --------------------------------------------------------

    y_pixel_ticks = np.array([
        63,
        120,
        178,
        236,
        294
    ], dtype=float)

    y_value_ticks = np.array([
        -4,
        -3,
        -2,
        -1,
        0
    ], dtype=float)

    # --------------------------------------------------------
    # Calculate calibration equations
    # --------------------------------------------------------

    x_slope, x_intercept = np.polyfit(
        x_pixel_ticks,
        x_value_ticks,
        1
    )

    y_slope, y_intercept = np.polyfit(
        y_pixel_ticks,
        y_value_ticks,
        1
    )

    # --------------------------------------------------------
    # Detect black pixels
    # --------------------------------------------------------

    gray = np.mean(arr, axis=2)

    black_pixels = gray < 100

    roi = black_pixels[
        y_top:y_bottom + 1,
        x_left:x_right + 1
    ]

    ys, xs = np.where(roi)

    xs = xs + x_left
    ys = ys + y_top

    # --------------------------------------------------------
    # Convert pixels to graph coordinates
    # --------------------------------------------------------

    times = (
        x_slope * xs
        + x_intercept
    )

    magnitudes = (
        y_slope * ys
        + y_intercept
    )

    digitized = pd.DataFrame({
        "pixel_x": xs,
        "pixel_y": ys,
        "time_seconds": times,
        "magnitude": magnitudes
    })

    return digitized


# ============================================================
# STAGE 2.1 — CREATE REPRESENTATIVE CURVE
# ============================================================

def create_representative_curve(digitized):

    curve = (
        digitized
        .groupby(
            "pixel_x",
            as_index=False
        )
        .agg({
            "time_seconds": "median",
            "magnitude": "median"
        })
    )

    curve = curve.sort_values(
        "pixel_x"
    ).reset_index(drop=True)

    return curve


# ============================================================
# STAGE 2.2 — CONTROLLED INTERPOLATION
# ============================================================

def interpolate_small_gaps(curve):

    curve = curve.copy()

    existing_pixels = (
        curve["pixel_x"]
        .astype(int)
        .to_numpy()
    )

    if len(existing_pixels) == 0:
        return curve

    min_pixel = existing_pixels.min()
    max_pixel = existing_pixels.max()

    full_pixels = np.arange(
        min_pixel,
        max_pixel + 1
    )

    curve = (
        curve
        .set_index("pixel_x")
        .reindex(full_pixels)
    )

    curve.index.name = "pixel_x"

    missing = curve["magnitude"].isna()

    groups = (
        missing
        .ne(missing.shift())
        .cumsum()
    )

    for _, indices in curve[
        missing
    ].groupby(groups):

        start = indices.index[0]
        end = indices.index[-1]

        gap_size = end - start + 1

        if gap_size <= MAX_INTERPOLATION_GAP:

            left = start - 1
            right = end + 1

            if (
                left in curve.index
                and right in curve.index
                and not pd.isna(
                    curve.loc[left, "magnitude"]
                )
                and not pd.isna(
                    curve.loc[right, "magnitude"]
                )
            ):

                gap_pixels = np.arange(
                    start,
                    end + 1
                )

                curve.loc[
                    start:end,
                    "magnitude"
                ] = np.interp(
                    gap_pixels,
                    [left, right],
                    [
                        curve.loc[
                            left,
                            "magnitude"
                        ],
                        curve.loc[
                            right,
                            "magnitude"
                        ]
                    ]
                )

                curve.loc[
                    start:end,
                    "time_seconds"
                ] = np.interp(
                    gap_pixels,
                    [left, right],
                    [
                        curve.loc[
                            left,
                            "time_seconds"
                        ],
                        curve.loc[
                            right,
                            "time_seconds"
                        ]
                    ]
                )

    return curve.reset_index()


# ============================================================
# STAGE 2.3 — GAP-AWARE SMOOTHING
# ============================================================

def smooth_segments(curve):

    curve = curve.copy()

    window = SMOOTH_WINDOW

    if window % 2 == 0:
        window += 1

    curve["magnitude_smooth"] = np.nan

    valid = curve["magnitude"].notna()

    groups = (
        valid
        .ne(valid.shift())
        .cumsum()
    )

    for _, segment in curve[
        valid
    ].groupby(groups):

        indices = segment.index

        curve.loc[
            indices,
            "magnitude_smooth"
        ] = (
            curve.loc[
                indices,
                "magnitude"
            ]
            .rolling(
                window=window,
                center=True,
                min_periods=1
            )
            .median()
        )

    return curve


# ============================================================
# COMPLETE RECONSTRUCTION
# ============================================================

def reconstruct_curve(digitized):

    curve = create_representative_curve(
        digitized
    )

    curve = interpolate_small_gaps(
        curve
    )

    curve = smooth_segments(
        curve
    )

    return curve


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_one_image(
    image_path,
    save_processed_plot=True
):

    digitized = digitize_png(
        image_path
    )

    if digitized.empty:
        raise RuntimeError(
            "No dark pixels detected in the plot area."
        )

    processed = reconstruct_curve(
        digitized
    )

    # --------------------------------------------------------
    # CSV output
    # --------------------------------------------------------

    csv_path = (
        image_path.parent
        / f"{image_path.stem}.csv"
    )

    processed.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Processed plot
    # --------------------------------------------------------

    png_path = None

    if save_processed_plot:

        png_path = (
            image_path.parent
            / f"{image_path.stem}_processed.png"
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.scatter(
            digitized["time_seconds"],
            digitized["magnitude"],
            s=1,
            alpha=0.08,
            label="Digitized pixels"
        )

        plt.plot(
            processed["time_seconds"],
            processed["magnitude"],
            linewidth=1,
            alpha=0.45,
            label="Reconstructed curve"
        )

        plt.plot(
            processed["time_seconds"],
            processed["magnitude_smooth"],
            linewidth=2,
            label="Continuous light curve"
        )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Standard Magnitude")

        plt.gca().invert_yaxis()

        plt.title(
            "MMT-9 — Reconstructed Folded Light Curve"
        )

        plt.legend()

        plt.grid(alpha=0.2)

        plt.tight_layout()

        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    return csv_path, png_path, len(digitized)


# ============================================================
# FIND MMT-9 FOLDed LIGHT CURVES
# ============================================================

def find_input_images(root_folder):

    root = Path(root_folder)

    pattern = re.compile(
        r"^\d+_F\.(png|jpg|jpeg)$",
        re.IGNORECASE
    )

    images = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if pattern.match(path.name):
            images.append(path)

    images.sort()

    return images


# ============================================================
# BATCH PROCESSING
# ============================================================

def batch_process(
    root_folder,
    logger
):

    images = find_input_images(
        root_folder
    )

    if not images:

        raise RuntimeError(
            "No *_F.png light curves were found."
        )

    logger(
        f"Found {len(images)} folded light curves."
    )

    logger("")

    successful = 0
    failed = 0

    for number, image_path in enumerate(
        images,
        start=1
    ):

        if stop_event.is_set():
            break

        logger("=" * 70)

        logger(
            f"[{number}/{len(images)}] "
            f"NORAD/TRACK: "
            f"{image_path.parent.parent.name}/"
            f"{image_path.parent.name}"
        )

        logger(
            f"Input: {image_path.name}"
        )

        try:

            csv_path, png_path, pixel_count = (
                process_one_image(
                    image_path
                )
            )

            successful += 1

            logger(
                f"✓ Extracted {pixel_count:,} pixels."
            )

            logger(
                f"✓ CSV: {csv_path.name}"
            )

            if png_path:
                logger(
                    f"✓ Plot: {png_path.name}"
                )

        except Exception as error:

            failed += 1

            logger(
                f"✗ FAILED: {error}"
            )

        logger("")

    logger("=" * 70)

    if stop_event.is_set():

        logger(
            "⏹ BATCH PROCESSING STOPPED."
        )

    else:

        logger(
            "✓ BATCH PROCESSING COMPLETE."
        )

    logger(
        f"Successful: {successful}"
    )

    logger(
        f"Failed: {failed}"
    )


# ============================================================
# GUI
# ============================================================

class App:

    def __init__(self, root):

        self.root = root

        root.title(
            "MMT-9 Batch Light Curve Processor"
        )

        root.geometry(
            "900x650"
        )

        self.folder_var = tk.StringVar()

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        tk.Label(
            root,
            text="MMT-9 Batch Light Curve Processor",
            font=("Arial", 20, "bold")
        ).pack(
            pady=(15, 5)
        )

        tk.Label(
            root,
            text=(
                "Processes all *_F.png folded light curves "
                "inside the selected MMT-9 folder."
            )
        ).pack(
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Folder selection
        # ----------------------------------------------------

        folder_frame = tk.Frame(root)

        folder_frame.pack(
            fill=tk.X,
            padx=20,
            pady=5
        )

        tk.Label(
            folder_frame,
            text="MMT-9 data folder:",
            width=20,
            anchor="w"
        ).pack(
            side=tk.LEFT
        )

        tk.Entry(
            folder_frame,
            textvariable=self.folder_var
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=10
        )

        tk.Button(
            folder_frame,
            text="Browse...",
            command=self.select_folder
        ).pack(
            side=tk.RIGHT
        )

        # ----------------------------------------------------
        # Buttons
        # ----------------------------------------------------

        button_frame = tk.Frame(root)

        button_frame.pack(
            pady=15
        )

        self.start_button = tk.Button(
            button_frame,
            text="START BATCH",
            width=18,
            height=2,
            command=self.start
        )

        self.start_button.pack(
            side=tk.LEFT,
            padx=10
        )

        self.stop_button = tk.Button(
            button_frame,
            text="STOP",
            width=18,
            height=2,
            command=self.stop,
            state=tk.DISABLED
        )

        self.stop_button.pack(
            side=tk.LEFT,
            padx=10
        )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        tk.Label(
            root,
            text="Activity Log",
            font=("Arial", 12, "bold")
        ).pack(
            anchor=tk.W,
            padx=20
        )

        self.log = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )

        self.log.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=(5, 20)
        )

    # ========================================================
    # LOG
    # ========================================================

    def write_log(self, text):

        self.log.insert(
            tk.END,
            text + "\n"
        )

        self.log.see(
            tk.END
        )

        self.root.update_idletasks()

    # ========================================================
    # SELECT FOLDER
    # ========================================================

    def select_folder(self):

        folder = filedialog.askdirectory(
            title="Select MMT-9 data folder"
        )

        if folder:
            self.folder_var.set(
                folder
            )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        folder = (
            self.folder_var
            .get()
            .strip()
        )

        if not folder:

            messagebox.showwarning(
                "No folder selected",
                "Please select the MMT-9 data folder."
            )

            return

        if not Path(folder).exists():

            messagebox.showerror(
                "Folder not found",
                "The selected folder does not exist."
            )

            return

        stop_event.clear()

        self.start_button.config(
            state=tk.DISABLED
        )

        self.stop_button.config(
            state=tk.NORMAL
        )

        self.log.delete(
            "1.0",
            tk.END
        )

        worker = threading.Thread(
            target=self.run_worker,
            args=(folder,),
            daemon=True
        )

        worker.start()

    # ========================================================
    # WORKER
    # ========================================================

    def run_worker(self, folder):

        try:

            self.root.after(
                0,
                self.write_log,
                "=" * 70
            )

            self.root.after(
                0,
                self.write_log,
                "MMT-9 BATCH LIGHT CURVE PROCESSOR"
            )

            self.root.after(
                0,
                self.write_log,
                "=" * 70
            )

            self.root.after(
                0,
                self.write_log,
                f"Root: {folder}"
            )

            # Thread-safe logger.
            def logger(message):

                self.root.after(
                    0,
                    self.write_log,
                    message
                )

            batch_process(
                folder,
                logger
            )

        except Exception as error:

            self.root.after(
                0,
                self.write_log,
                f"ERROR: {error}"
            )

        finally:

            self.root.after(
                0,
                self.finished
            )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        stop_event.set()

        self.stop_button.config(
            state=tk.DISABLED
        )

        self.write_log(
            "⏹ Stop requested. Finishing current image..."
        )

    # ========================================================
    # FINISHED
    # ========================================================

    def finished(self):

        self.start_button.config(
            state=tk.NORMAL
        )

        self.stop_button.config(
            state=tk.DISABLED
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.mainloop()
