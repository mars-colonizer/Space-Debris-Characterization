import csv
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox


# ============================================================
# MMT-9 LIGHT CURVE LOCATION SCANNER
#
# Folder structure expected:
#
# data/
# └── mmt9/
#     ├── 60424/
#     │   ├── 29185867/
#     │   │   ├── 29185867_D.png
#     │   │   ├── 29185867_F.png
#     │   │   ├── 29185867_L.png
#     │   │   ├── 29185867_M.png
#     │   │   ├── 29185867_P.png
#     │   │   ├── 29185867_R.png
#     │   │   └── ...
#     │   └── ...
#     └── 17360/
#         └── ...
#
# Output:
#
# NORAD ID, TRACK ID, D, F, L, M, P, R
#
# ============================================================


# Plot type -> CSV column
PLOT_TYPES = {
    "D": "D",  # Distance
    "F": "F",  # Folded light curve
    "L": "L",  # Raw light curve
    "M": "M",  # PDM plot
    "P": "P",  # Lomb-Scargle periodogram
    "R": "R",  # Raw light curve with standard magnitude
}


# ============================================================
# FIND PLOT FILES
# ============================================================

def scan_mmt9(root_folder, log_callback=None):

    root = Path(root_folder)

    if not root.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{root}"
        )

    rows = []

    norad_folders = [
        folder
        for folder in root.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    ]

    # Sort numerically by NORAD ID
    norad_folders.sort(
        key=lambda x: int(x.name)
    )

    if log_callback:
        log_callback(
            f"Found {len(norad_folders)} NORAD folders."
        )

    for norad_folder in norad_folders:

        norad_id = norad_folder.name

        track_folders = [
            folder
            for folder in norad_folder.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        ]

        track_folders.sort(
            key=lambda x: int(x.name)
        )

        if log_callback:
            log_callback(
                f"NORAD {norad_id}: "
                f"{len(track_folders)} track folders"
            )

        for track_folder in track_folders:

            track_id = track_folder.name

            # Default values are empty.
            # This handles incomplete/empty track folders.
            paths = {
                "D": "",
                "F": "",
                "L": "",
                "M": "",
                "P": "",
                "R": "",
            }

            # Look only at files directly inside the track folder.
            for file in track_folder.iterdir():

                if not file.is_file():
                    continue

                # Expected naming:
                # TRACKID_D.png
                # TRACKID_F.png
                # etc.
                #
                # We also allow jpg/jpeg in case the format changes.
                match = re.match(
                    rf"^{re.escape(track_id)}_([DFLMPR])"
                    rf"\.(png|jpg|jpeg)$",
                    file.name,
                    re.IGNORECASE,
                )

                if not match:
                    continue

                plot_type = match.group(1).upper()

                if plot_type in paths:
                    paths[plot_type] = str(
                        file.resolve()
                    )

            # Add one row per track, even if the folder is empty.
            rows.append([
                norad_id,
                track_id,
                paths["D"],
                paths["F"],
                paths["L"],
                paths["M"],
                paths["P"],
                paths["R"],
            ])

    return rows


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(rows, output_file):

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    headers = [
        "NORAD ID",
        "TRACK ID",
        "D",
        "F",
        "L",
        "M",
        "P",
        "R",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)
        writer.writerows(rows)


# ============================================================
# GUI
# ============================================================

class App:

    def __init__(self, root):

        self.root = root

        root.title(
            "MMT-9 File Location Scanner"
        )

        root.geometry(
            "850x600"
        )

        self.input_folder = tk.StringVar()

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        tk.Label(
            root,
            text="MMT-9 File Location Scanner",
            font=("Arial", 20, "bold")
        ).pack(
            pady=(15, 10)
        )

        tk.Label(
            root,
            text=(
                "Select the folder containing the NORAD ID folders."
            )
        ).pack(
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # Input folder
        # ----------------------------------------------------

        input_frame = tk.Frame(root)

        input_frame.pack(
            fill=tk.X,
            padx=20,
            pady=5
        )

        tk.Label(
            input_frame,
            text="MMT-9 data folder:",
            width=20,
            anchor="w"
        ).pack(
            side=tk.LEFT
        )

        tk.Entry(
            input_frame,
            textvariable=self.input_folder
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=10
        )

        tk.Button(
            input_frame,
            text="Browse...",
            command=self.select_folder
        ).pack(
            side=tk.RIGHT
        )

        # ----------------------------------------------------
        # Scan button
        # ----------------------------------------------------

        self.scan_button = tk.Button(
            root,
            text="SCAN & CREATE CSV",
            width=25,
            height=2,
            command=self.run_scan
        )

        self.scan_button.pack(
            pady=15
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

        self.log = tk.Text(
            root,
            height=20,
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
            self.input_folder.set(
                folder
            )

    # ========================================================
    # RUN SCAN
    # ========================================================

    def run_scan(self):

        folder = (
            self.input_folder
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

        self.scan_button.config(
            state=tk.DISABLED
        )

        self.log.delete(
            "1.0",
            tk.END
        )

        try:

            self.write_log(
                "=" * 70
            )

            self.write_log(
                "MMT-9 FILE LOCATION SCANNER"
            )

            self.write_log(
                "=" * 70
            )

            self.write_log(
                f"Root folder: {folder}"
            )

            self.write_log("")

            rows = scan_mmt9(
                folder,
                self.write_log
            )

            # ------------------------------------------------
            # Output location
            # ------------------------------------------------

            root_path = Path(folder)

            output_file = (
                root_path / "mmt9_file_locations.csv"
            )

            write_csv(
                rows,
                output_file
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_tracks = len(rows)

            counts = {
                "D": 0,
                "F": 0,
                "L": 0,
                "M": 0,
                "P": 0,
                "R": 0,
            }

            for row in rows:

                for index, plot_type in enumerate(
                    ["D", "F", "L", "M", "P", "R"],
                    start=2
                ):

                    if row[index]:
                        counts[plot_type] += 1

            self.write_log("")
            self.write_log("=" * 70)
            self.write_log("SCAN COMPLETE")
            self.write_log("=" * 70)

            self.write_log(
                f"Tracks found: {total_tracks}"
            )

            self.write_log(
                f"D - Distance: {counts['D']}"
            )

            self.write_log(
                f"F - Folded light curve: {counts['F']}"
            )

            self.write_log(
                f"L - Raw light curve: {counts['L']}"
            )

            self.write_log(
                f"M - PDM plot: {counts['M']}"
            )

            self.write_log(
                f"P - Lomb-Scargle periodogram: {counts['P']}"
            )

            self.write_log(
                f"R - Raw light curve + standard magnitude: "
                f"{counts['R']}"
            )

            self.write_log("")
            self.write_log(
                f"CSV created:\n{output_file}"
            )

            messagebox.showinfo(
                "Complete",
                (
                    f"CSV created successfully.\n\n"
                    f"Tracks: {total_tracks}\n\n"
                    f"{output_file}"
                )
            )

        except Exception as e:

            self.write_log("")
            self.write_log(
                f"ERROR: {e}"
            )

            messagebox.showerror(
                "Error",
                str(e)
            )

        finally:

            self.scan_button.config(
                state=tk.NORMAL
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = App(root)

    root.mainloop()
