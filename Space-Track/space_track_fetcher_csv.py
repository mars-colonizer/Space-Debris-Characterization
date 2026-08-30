import csv
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.space-track.org"

LOGIN_URL = f"{BASE_URL}/ajaxauth/login"

GP_URL = f"{BASE_URL}/basicspacedata/query/class/gp/"

# One common CSV file for all NORAD IDs
OUTPUT_DIR = Path("data") / "space_track"
OUTPUT_CSV = OUTPUT_DIR / "tle_data.csv"

REQUEST_TIMEOUT = 60
REQUEST_DELAY = 1.0


# ============================================================
# STOP CONTROL
# ============================================================

stop_event = threading.Event()


# ============================================================
# LOGGER
# ============================================================

class Logger:

    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.after(
            0,
            self._write,
            message
        )

    def _write(self, message):
        self.widget.insert(
            tk.END,
            message + "\n"
        )
        self.widget.see(tk.END)


# ============================================================
# READ NORAD IDs FROM CSV
# ============================================================

def read_norad_ids(csv_path):

    norad_ids = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "CSV file does not contain a header."
            )

        norad_column = None

        for column in reader.fieldnames:

            normalized = (
                column.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized in {
                "norad_id",
                "noradid",
                "catalogue_id",
                "catalog_id",
            }:
                norad_column = column
                break

        if norad_column is None:
            raise ValueError(
                "CSV must contain a NORAD_ID column."
            )

        for row in reader:

            value = (
                row.get(norad_column) or ""
            ).strip()

            if not value:
                continue

            # Handle values such as 60424.0
            if value.endswith(".0"):
                value = value[:-2]

            if value.isdigit() and value not in norad_ids:
                norad_ids.append(value)

    return norad_ids


# ============================================================
# SPACE-TRACK LOGIN
# ============================================================

def login(session, username, password, logger):

    logger.write("")
    logger.write("=" * 60)
    logger.write("LOGGING INTO SPACE-TRACK")
    logger.write("=" * 60)

    response = session.post(
        LOGIN_URL,
        data={
            "identity": username,
            "password": password,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Login HTTP error: {response.status_code}"
        )

    if "chocolatechip" not in session.cookies:
        raise RuntimeError(
            "Space-Track did not return an authentication cookie."
        )

    logger.write(
        "✓ Space-Track authentication successful."
    )


# ============================================================
# FETCH TLE
# ============================================================

def fetch_tle(session, norad_id):

    url = (
        GP_URL
        + f"norad_cat_id/{norad_id}/"
        + "format/tle"
    )

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}"
        )

    text = response.text.strip()

    if not text:
        raise RuntimeError(
            "Space-Track returned an empty response."
        )

    return text


# ============================================================
# CREATE / UPDATE COMMON TLE CSV
# ============================================================

def ensure_output_csv():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not OUTPUT_CSV.exists():

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "NORAD_ID",
                "TLE_LINE_1",
                "TLE_LINE_2"
            ])


def save_tle_to_csv(norad_id, tle_text):

    ensure_output_csv()

    lines = [
        line.strip()
        for line in tle_text.splitlines()
        if line.strip()
    ]

    # Space-Track's TLE response normally contains:
    #   Line 1
    #   Line 2
    #
    # Some responses can also contain an object name.
    # Keep only the two actual TLE lines.

    tle_lines = [
        line
        for line in lines
        if line.startswith("1 ")
        or line.startswith("2 ")
    ]

    line1 = ""
    line2 = ""

    for line in tle_lines:

        if line.startswith("1 "):
            line1 = line

        elif line.startswith("2 "):
            line2 = line

    if not line1 or not line2:
        raise RuntimeError(
            "Could not identify both TLE lines."
        )

    # Read existing records.
    rows = []

    with open(
        OUTPUT_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["NORAD_ID"] != str(norad_id):
                rows.append([
                    row["NORAD_ID"],
                    row["TLE_LINE_1"],
                    row["TLE_LINE_2"]
                ])

    # Add / replace the record for this NORAD ID.
    rows.append([
        str(norad_id),
        line1,
        line2
    ])

    # Rewrite the common CSV.
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "NORAD_ID",
            "TLE_LINE_1",
            "TLE_LINE_2"
        ])

        writer.writerows(rows)

    return line1, line2


# ============================================================
# FETCHER
# ============================================================

def run_fetcher(
    csv_path,
    username,
    password,
    logger
):

    try:

        norad_ids = read_norad_ids(
            csv_path
        )

        if not norad_ids:
            raise ValueError(
                "No NORAD IDs were found in the CSV."
            )

        logger.write(
            f"Loaded {len(norad_ids)} NORAD IDs."
        )

        ensure_output_csv()

        session = requests.Session()

        session.headers.update({
            "User-Agent":
                "Space-Debris-Classification/1.0",
            "Accept":
                "*/*",
        })

        # Login once.
        login(
            session,
            username,
            password,
            logger
        )

        successful = 0
        failed = 0

        for number, norad_id in enumerate(
            norad_ids,
            start=1
        ):

            if stop_event.is_set():
                break

            logger.write("")
            logger.write("=" * 60)
            logger.write(
                f"OBJECT [{number}/{len(norad_ids)}] "
                f"NORAD {norad_id}"
            )
            logger.write("=" * 60)

            try:

                tle_text = fetch_tle(
                    session,
                    norad_id
                )

                line1, line2 = save_tle_to_csv(
                    norad_id,
                    tle_text
                )

                successful += 1

                logger.write(
                    "✓ TLE saved to common CSV."
                )

                logger.write(
                    f"  NORAD ID: {norad_id}"
                )

                logger.write(
                    f"  TLE 1: {line1}"
                )

                logger.write(
                    f"  TLE 2: {line2}"
                )

            except Exception as e:

                failed += 1

                logger.write(
                    f"✗ Failed NORAD {norad_id}: {e}"
                )

            # Small delay between requests.
            if number < len(norad_ids):

                for _ in range(10):

                    if stop_event.is_set():
                        break

                    time.sleep(
                        REQUEST_DELAY / 10
                    )

        logger.write("")
        logger.write("=" * 60)

        if stop_event.is_set():

            logger.write(
                "⏹ FETCHER STOPPED"
            )

        else:

            logger.write(
                "✓ FETCH COMPLETE"
            )

        logger.write(
            f"Successful: {successful}"
        )

        logger.write(
            f"Failed: {failed}"
        )

        logger.write(
            f"Output: {OUTPUT_CSV}"
        )

    except Exception as e:

        logger.write("")
        logger.write("=" * 60)
        logger.write("ERROR")
        logger.write("=" * 60)
        logger.write(str(e))

    finally:

        root.after(
            0,
            fetch_finished
        )


# ============================================================
# GUI FUNCTIONS
# ============================================================

def choose_csv():

    path = filedialog.askopenfilename(
        title="Select NORAD ID CSV",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    if path:
        csv_path_var.set(path)


def start_fetch():

    csv_path = (
        csv_path_var.get().strip()
    )

    username = (
        username_var.get().strip()
    )

    password = (
        password_var.get()
    )

    if not csv_path:

        messagebox.showwarning(
            "Missing CSV",
            "Please select a NORAD ID CSV file."
        )

        return

    if not Path(csv_path).exists():

        messagebox.showerror(
            "File not found",
            "The selected CSV file does not exist."
        )

        return

    if not username:

        messagebox.showwarning(
            "Missing username",
            "Enter your Space-Track username."
        )

        return

    if not password:

        messagebox.showwarning(
            "Missing password",
            "Enter your Space-Track password."
        )

        return

    stop_event.clear()

    start_button.config(
        state=tk.DISABLED
    )

    stop_button.config(
        state=tk.NORMAL
    )

    browse_button.config(
        state=tk.DISABLED
    )

    log_box.delete(
        "1.0",
        tk.END
    )

    worker = threading.Thread(
        target=run_fetcher,
        args=(
            csv_path,
            username,
            password,
            logger
        ),
        daemon=True
    )

    worker.start()


def stop_fetch():

    stop_event.set()

    stop_button.config(
        state=tk.DISABLED
    )

    logger.write("")
    logger.write(
        "⏹ Stop requested. "
        "Finishing the current request..."
    )


def fetch_finished():

    start_button.config(
        state=tk.NORMAL
    )

    stop_button.config(
        state=tk.DISABLED
    )

    browse_button.config(
        state=tk.NORMAL
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Space-Track TLE Fetcher"
)

root.geometry(
    "900x700"
)

csv_path_var = tk.StringVar()
username_var = tk.StringVar()
password_var = tk.StringVar()


tk.Label(
    root,
    text="Space-Track TLE Fetcher",
    font=("Arial", 20, "bold"),
).pack(
    pady=(15, 10)
)


# ------------------------------------------------------------
# CSV selection
# ------------------------------------------------------------

csv_frame = tk.Frame(root)

csv_frame.pack(
    fill=tk.X,
    padx=20,
    pady=5
)

tk.Label(
    csv_frame,
    text="NORAD CSV:",
    width=15,
    anchor="w",
).pack(
    side=tk.LEFT
)

tk.Entry(
    csv_frame,
    textvariable=csv_path_var,
).pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    padx=10
)

browse_button = tk.Button(
    csv_frame,
    text="Browse...",
    command=choose_csv
)

browse_button.pack(
    side=tk.RIGHT
)


# ------------------------------------------------------------
# Username
# ------------------------------------------------------------

username_frame = tk.Frame(root)

username_frame.pack(
    fill=tk.X,
    padx=20,
    pady=5
)

tk.Label(
    username_frame,
    text="Username:",
    width=15,
    anchor="w",
).pack(
    side=tk.LEFT
)

tk.Entry(
    username_frame,
    textvariable=username_var,
).pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True
)


# ------------------------------------------------------------
# Password
# ------------------------------------------------------------

password_frame = tk.Frame(root)

password_frame.pack(
    fill=tk.X,
    padx=20,
    pady=5
)

tk.Label(
    password_frame,
    text="Password:",
    width=15,
    anchor="w",
).pack(
    side=tk.LEFT
)

tk.Entry(
    password_frame,
    textvariable=password_var,
    show="*",
).pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True
)


# ------------------------------------------------------------
# Buttons
# ------------------------------------------------------------

button_frame = tk.Frame(root)

button_frame.pack(
    pady=15
)

start_button = tk.Button(
    button_frame,
    text="START",
    width=15,
    command=start_fetch
)

start_button.pack(
    side=tk.LEFT,
    padx=10
)

stop_button = tk.Button(
    button_frame,
    text="STOP",
    width=15,
    command=stop_fetch,
    state=tk.DISABLED
)

stop_button.pack(
    side=tk.LEFT,
    padx=10
)


# ------------------------------------------------------------
# Activity log
# ------------------------------------------------------------

tk.Label(
    root,
    text="Activity Log",
    font=("Arial", 12, "bold"),
).pack(
    anchor=tk.W,
    padx=20
)

log_box = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Consolas", 10)
)

log_box.pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=(5, 20)
)


logger = Logger(log_box)


# ============================================================
# START
# ============================================================

root.mainloop()
