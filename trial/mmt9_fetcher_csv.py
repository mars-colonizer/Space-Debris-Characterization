from pathlib import Path
from urllib.parse import urljoin
import csv
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

MMT9_URL = "http://mmt.favor2.info/satellites"

BASE_DATA_DIR = Path("data") / "mmt9"

MAX_PAGE_RETRIES = 5
PAGE_RETRY_DELAY = 5

PRODUCT_LABELS = {
    "L", "S", "D", "P", "I", "F", "R", "M", "T"
}


# ============================================================
# GLOBAL STOP EVENT
# ============================================================

stop_event = threading.Event()


# ============================================================
# GUI LOGGING
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
        self.widget.insert(tk.END, message + "\n")
        self.widget.see(tk.END)


# ============================================================
# CSV READER
# ============================================================

def read_norad_ids(csv_path):
    """
    Read NORAD IDs from a CSV file.

    The CSV must contain a column named NORAD_ID.
    Matching is case-insensitive and whitespace is ignored.
    """

    norad_ids = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")

        # Find NORAD_ID column case-insensitively.
        norad_column = None

        for column in reader.fieldnames:

            normalized = column.strip().lower()

            if normalized in {
                "norad_id",
                "norad id",
                "noradid",
                "catalogue id",
                "catalogue_id"
            }:
                norad_column = column
                break

        if norad_column is None:

            raise ValueError(
                "CSV must contain a NORAD_ID column."
            )

        for row in reader:

            value = (
                row.get(norad_column)
                or ""
            ).strip()

            if not value:
                continue

            # Remove Excel-style .0 from numeric IDs.
            if re.fullmatch(r"\d+\.0", value):
                value = value[:-2]

            # Keep only valid numeric NORAD IDs.
            if re.fullmatch(r"\d+", value):

                if value not in norad_ids:
                    norad_ids.append(value)

    return norad_ids


# ============================================================
# OPEN MMT-9
# ============================================================

def open_mmt9(page, logger):

    for attempt in range(1, MAX_PAGE_RETRIES + 1):

        if stop_event.is_set():
            return False

        logger.write(
            f"Opening MMT-9 "
            f"(attempt {attempt}/{MAX_PAGE_RETRIES})..."
        )

        try:

            page.goto(
                MMT9_URL,
                wait_until="domcontentloaded",
                timeout=60000
            )

            page.wait_for_timeout(3000)

            body = page.locator("body")

            if body.count() == 0:
                logger.write("  Blank page.")
                continue

            text = body.inner_text(
                timeout=10000
            ).strip()

            if text:

                logger.write("✓ MMT-9 loaded.")
                return True

            logger.write("⚠ MMT-9 returned a blank page.")

        except Exception as e:

            logger.write(f"⚠ Error: {e}")

        if attempt < MAX_PAGE_RETRIES:

            for _ in range(PAGE_RETRY_DELAY):

                if stop_event.is_set():
                    return False

                time.sleep(1)

    return False


# ============================================================
# FIND CATALOGUE INPUT
# ============================================================

def find_catalogue_input(page):

    inputs = page.locator("input")

    for i in range(inputs.count()):

        element = inputs.nth(i)

        if not element.is_visible():
            continue

        placeholder = (
            element.get_attribute("placeholder")
            or ""
        ).lower()

        name = (
            element.get_attribute("name")
            or ""
        ).lower()

        element_id = (
            element.get_attribute("id")
            or ""
        ).lower()

        if (
            "catalogue" in placeholder
            or "catalog" in placeholder
            or "catalogue" in name
            or "catalog" in name
            or "catalogue" in element_id
            or "catalog" in element_id
        ):
            return element

    for i in range(inputs.count()):

        element = inputs.nth(i)

        if not element.is_visible():
            continue

        input_type = (
            element.get_attribute("type")
            or "text"
        ).lower()

        if input_type in ("text", "search"):
            return element

    return None


# ============================================================
# SEARCH NORAD
# ============================================================

def search_norad(page, norad_id, logger):

    if stop_event.is_set():
        return False

    logger.write(
        f"Searching NORAD ID {norad_id}..."
    )

    catalogue_input = find_catalogue_input(page)

    if catalogue_input is None:
        raise RuntimeError(
            "Could not find Catalogue ID input."
        )

    catalogue_input.fill(norad_id)

    search_button = None

    buttons = page.locator("button")

    for i in range(buttons.count()):

        button = buttons.nth(i)

        if not button.is_visible():
            continue

        if (
            button.inner_text()
            .strip()
            .lower()
            == "search"
        ):
            search_button = button
            break

    if search_button is None:

        submits = page.locator(
            'input[type="submit"]'
        )

        for i in range(submits.count()):

            button = submits.nth(i)

            if not button.is_visible():
                continue

            value = (
                button.get_attribute("value")
                or ""
            ).lower()

            if "search" in value:

                search_button = button
                break

    if search_button is None:
        raise RuntimeError(
            "Could not find Search button."
        )

    search_button.click()

    page.wait_for_timeout(3000)

    logger.write(
        f"✓ NORAD {norad_id} found."
    )

    return True


# ============================================================
# FIND TRACK TABLE
# ============================================================

def find_track_table(page):

    tables = page.locator("table")

    for table_index in range(tables.count()):

        table = tables.nth(table_index)

        rows = table.locator("tr")

        for row_index in range(rows.count()):

            text = (
                rows.nth(row_index)
                .inner_text()
                .strip()
                .lower()
            )

            if (
                "track id" in text
                and "start" in text
                and "duration" in text
                and "records" in text
            ):
                return table

    return None


# ============================================================
# EXTRACT TRACK ID
# ============================================================

def extract_track_id(row):

    links = row.locator("a")

    for i in range(links.count()):

        text = links.nth(i).inner_text().strip()

        if re.fullmatch(r"\d+", text):
            return text

    match = re.search(
        r"\b\d{6,}\b",
        row.inner_text()
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# EXTRACT PERIOD
# ============================================================

def extract_period(row):

    text = row.inner_text()

    if re.search(
        r"\bAperiodic\b",
        text,
        re.IGNORECASE
    ):
        return None

    match = re.search(
        r"Period:\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return None


# ============================================================
# EXTRACT PRODUCT LINKS
# ============================================================

def extract_product_links(row):

    links = row.locator("a")

    products = []

    for i in range(links.count()):

        link = links.nth(i)

        label = link.inner_text().strip().upper()
        href = link.get_attribute("href")

        if label in PRODUCT_LABELS:

            products.append(
                {
                    "label": label,
                    "href": href
                }
            )

    return products


# ============================================================
# EXTRACT PERIODIC TRACKS
# ============================================================

def extract_periodic_tracks(table, logger):

    rows = table.locator("tr")

    tracks = []

    for row_index in range(rows.count()):

        if stop_event.is_set():
            break

        row = rows.nth(row_index)

        text = row.inner_text().strip()

        if not text:
            continue

        if "Track ID" in text:
            continue

        track_id = extract_track_id(row)

        if not track_id:
            continue

        period = extract_period(row)

        if period is None:

            logger.write(
                f"  {track_id}: Aperiodic → SKIP"
            )

            continue

        products = extract_product_links(row)

        logger.write(
            f"  {track_id}: "
            f"Period = {period:.2f} s | "
            f"Products = "
            f"{','.join(p['label'] for p in products)}"
        )

        tracks.append(
            {
                "track_id": track_id,
                "period": period,
                "row_index": row_index,
                "products": products
            }
        )

    logger.write(
        f"  Periodic tracks: {len(tracks)}"
    )

    return tracks


# ============================================================
# DOWNLOAD ONE PRODUCT
# ============================================================

def download_product(
    page,
    product,
    track_id,
    track_dir,
    logger
):

    if stop_event.is_set():
        return None

    label = product["label"]
    href = product["href"]

    if not href:
        logger.write(
            f"    {label}: no link"
        )
        return None

    product_page = page.context.new_page()

    try:

        # Open the object page in the new browser tab.
        # This keeps the same browser context/session.
        product_page.goto(
            page.url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        product_page.wait_for_timeout(500)

        # Locate the exact product link.
        locator = product_page.locator(
            f'a[href="{href}"]'
        )

        if locator.count() == 0:

            locator = product_page.get_by_text(
                label,
                exact=True
            )

        if locator.count() == 0:

            logger.write(
                f"    {label}: link not found"
            )
            return None

        link = locator.first

        # Force same-tab navigation.
        link.evaluate(
            """
            element => {
                element.target = "_self";
            }
            """
        )

        logger.write(
            f"    Downloading {label}..."
        )

        link.click()

        product_page.wait_for_load_state(
            "domcontentloaded",
            timeout=60000
        )

        product_page.wait_for_timeout(1500)

        # MMT-9 products are displayed as images.
        images = product_page.locator("img")

        if images.count() == 0:

            logger.write(
                f"    ⚠ {label}: image not found"
            )
            return None

        image = images.first

        image.wait_for(
            state="visible",
            timeout=15000
        )

        filename = (
            f"{track_id}_{label}.png"
        )

        destination = (
            track_dir / filename
        )

        image.screenshot(
            path=str(destination)
        )

        logger.write(
            f"    ✓ {filename}"
        )

        return filename

    except Exception as e:

        logger.write(
            f"    ✗ {label}: {e}"
        )

        return None

    finally:

        product_page.close()


# ============================================================
# DOWNLOAD ONE TRACK
# ============================================================

def download_track(
    page,
    track,
    object_dir,
    logger
):

    if stop_event.is_set():
        return []

    track_id = track["track_id"]

    track_dir = (
        object_dir / track_id
    )

    track_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.write(
        f"  TRACK {track_id} "
        f"(Period = {track['period']:.2f} s)"
    )

    downloaded = []

    for product in track["products"]:

        if stop_event.is_set():
            break

        filename = download_product(
            page,
            product,
            track_id,
            track_dir,
            logger
        )

        if filename:
            downloaded.append(filename)

        time.sleep(0.5)

    return downloaded


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    object_dir,
    records
):

    path = object_dir / "manifest.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "norad_id",
                "track_id",
                "period_sec",
                "downloaded_products"
            ]
        )

        writer.writeheader()

        writer.writerows(records)


# ============================================================
# PROCESS ONE NORAD ID
# ============================================================

def process_norad(
    page,
    norad_id,
    logger
):

    if stop_event.is_set():
        return

    object_dir = (
        BASE_DATA_DIR / norad_id
    )

    object_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.write("")
    logger.write("=" * 60)
    logger.write(
        f"PROCESSING NORAD {norad_id}"
    )
    logger.write("=" * 60)

    # --------------------------------------------------------
    # Open database
    # --------------------------------------------------------

    if not open_mmt9(page, logger):

        if not stop_event.is_set():

            logger.write(
                f"✗ Could not load MMT-9 for {norad_id}"
            )

        return

    # --------------------------------------------------------
    # Search object
    # --------------------------------------------------------

    search_norad(
        page,
        norad_id,
        logger
    )

    if stop_event.is_set():
        return

    # --------------------------------------------------------
    # Find track table
    # --------------------------------------------------------

    table = find_track_table(page)

    if table is None:

        logger.write(
            f"✗ Track table not found for {norad_id}"
        )

        return

    # --------------------------------------------------------
    # Extract periodic tracks
    # --------------------------------------------------------

    tracks = extract_periodic_tracks(
        table,
        logger
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    manifest_records = []

    for number, track in enumerate(
        tracks,
        start=1
    ):

        if stop_event.is_set():
            break

        logger.write(
            f"  [{number}/{len(tracks)}]"
        )

        downloaded = download_track(
            page,
            track,
            object_dir,
            logger
        )

        manifest_records.append(
            {
                "norad_id": norad_id,
                "track_id": track["track_id"],
                "period_sec": track["period"],
                "downloaded_products":
                    ";".join(downloaded)
            }
        )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    if manifest_records:

        save_manifest(
            object_dir,
            manifest_records
        )

        logger.write(
            f"✓ Manifest saved for {norad_id}"
        )

    logger.write(
        f"✓ Finished NORAD {norad_id}"
    )


# ============================================================
# WORKER THREAD
# ============================================================

def run_fetcher(csv_path, logger):

    try:

        norad_ids = read_norad_ids(
            csv_path
        )

        if not norad_ids:

            raise ValueError(
                "No NORAD IDs found in CSV."
            )

        logger.write(
            f"Loaded {len(norad_ids)} NORAD IDs."
        )

        logger.write(
            "NORAD IDs: "
            + ", ".join(norad_ids)
        )

        with sync_playwright() as p:

            browser = p.chromium.launch(
                headless=False
            )

            context = browser.new_context(
                ignore_https_errors=True,
                accept_downloads=True
            )

            page = context.new_page()

            try:

                for number, norad_id in enumerate(
                    norad_ids,
                    start=1
                ):

                    if stop_event.is_set():
                        break

                    logger.write("")
                    logger.write(
                        f"OBJECT [{number}/{len(norad_ids)}]"
                    )

                    try:

                        process_norad(
                            page,
                            norad_id,
                            logger
                        )

                    except Exception as e:

                        logger.write(
                            f"✗ Error processing "
                            f"{norad_id}: {e}"
                        )

                        # Continue to the next object.
                        continue

            finally:

                context.close()
                browser.close()

        if stop_event.is_set():

            logger.write("")
            logger.write(
                "⏹ Process stopped by user."
            )

        else:

            logger.write("")
            logger.write(
                "=" * 60
            )
            logger.write(
                "✓ ALL OBJECTS COMPLETE"
            )
            logger.write(
                "=" * 60
            )

    except Exception as e:

        logger.write("")
        logger.write(
            f"ERROR: {e}"
        )

    finally:

        root.after(
            0,
            fetch_finished
        )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "MMT-9 Data Fetcher"
)

root.geometry(
    "900x650"
)

selected_csv = tk.StringVar(
    value=""
)

logger = None


def choose_csv():

    path = filedialog.askopenfilename(
        title="Select NORAD ID CSV",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )

    if path:

        selected_csv.set(path)


def start_fetch():

    csv_path = selected_csv.get().strip()

    if not csv_path:

        messagebox.showwarning(
            "No CSV selected",
            "Please select the NORAD ID CSV file first."
        )

        return

    if not Path(csv_path).exists():

        messagebox.showerror(
            "File not found",
            "The selected CSV file does not exist."
        )

        return

    # Prevent starting two fetchers.
    if start_button["state"] == tk.DISABLED:
        return

    stop_event.clear()

    start_button.config(
        state=tk.DISABLED
    )

    stop_button.config(
        state=tk.NORMAL
    )

    log_box.delete(
        "1.0",
        tk.END
    )

    worker = threading.Thread(
        target=run_fetcher,
        args=(csv_path, logger),
        daemon=True
    )

    worker.start()


def stop_fetch():

    stop_event.set()

    stop_button.config(
        state=tk.DISABLED
    )

    logger.write(
        "⏹ Stop requested. "
        "The current operation will finish, "
        "then the fetcher will stop."
    )


def fetch_finished():

    start_button.config(
        state=tk.NORMAL
    )

    stop_button.config(
        state=tk.DISABLED
    )


# ============================================================
# GUI LAYOUT
# ============================================================

title_label = tk.Label(
    root,
    text="MMT-9 Data Fetcher",
    font=("Arial", 20, "bold")
)

title_label.pack(
    pady=(15, 10)
)


file_frame = tk.Frame(root)

file_frame.pack(
    fill=tk.X,
    padx=20,
    pady=10
)


tk.Label(
    file_frame,
    text="NORAD ID CSV:"
).pack(
    side=tk.LEFT
)


csv_entry = tk.Entry(
    file_frame,
    textvariable=selected_csv
)

csv_entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    padx=10
)


browse_button = tk.Button(
    file_frame,
    text="Browse...",
    command=choose_csv
)

browse_button.pack(
    side=tk.RIGHT
)


button_frame = tk.Frame(root)

button_frame.pack(
    pady=10
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


tk.Label(
    root,
    text="Activity Log",
    font=("Arial", 12, "bold")
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
# START GUI
# ============================================================

root.mainloop()
