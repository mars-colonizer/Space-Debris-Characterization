from pathlib import Path
from urllib.parse import urljoin
import csv
import re
import time

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# CONFIGURATION
# ============================================================

NORAD_ID = "60424"

MMT9_URL = "http://mmt.favor2.info/satellites"

DATA_DIR = Path("data") / "mmt9" / NORAD_ID

MAX_PAGE_RETRIES = 5
PAGE_RETRY_DELAY = 5

# Products we are interested in.
# MMT-9 displays these as links on periodic tracks.
PRODUCT_LABELS = {
    "L",
    "S",
    "D",
    "P",
    "I",
    "F",
    "R",
    "M",
    "T",
}


# ============================================================
# OPEN MMT-9
# ============================================================

def open_mmt9(page):
    """
    Open MMT-9 with retries.

    MMT-9 occasionally returns a blank page, so we don't
    assume that a successful HTTP navigation means the site
    actually loaded.
    """

    for attempt in range(1, MAX_PAGE_RETRIES + 1):

        print(
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
                print("  No <body> element.")
                continue

            text = body.inner_text(
                timeout=10000
            ).strip()

            if len(text) > 0:

                print("✓ MMT-9 loaded.")
                return True

            print("⚠ MMT-9 returned a blank page.")

        except Exception as e:

            print("⚠ Error loading MMT-9:")
            print(f"  {e}")

        if attempt < MAX_PAGE_RETRIES:

            print(
                f"  Retrying in {PAGE_RETRY_DELAY} seconds..."
            )

            time.sleep(PAGE_RETRY_DELAY)

    return False


# ============================================================
# FIND CATALOGUE ID INPUT
# ============================================================

def find_catalogue_input(page):

    inputs = page.locator("input")

    # Try to identify the input from its attributes.
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

    # Fallback: visible text/search input.
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

def search_norad(page, norad_id):

    print()
    print("=" * 60)
    print(f"SEARCHING FOR NORAD ID {norad_id}")
    print("=" * 60)

    catalogue_input = find_catalogue_input(page)

    if catalogue_input is None:

        raise RuntimeError(
            "Could not find the Catalogue ID input."
        )

    catalogue_input.fill(norad_id)

    print("✓ NORAD ID entered.")

    # --------------------------------------------------------
    # Find Search button
    # --------------------------------------------------------

    search_button = None

    buttons = page.locator("button")

    for i in range(buttons.count()):

        button = buttons.nth(i)

        if not button.is_visible():
            continue

        text = button.inner_text().strip().lower()

        if text == "search":

            search_button = button
            break

    # Older HTML form fallback.
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
            ).strip().lower()

            if "search" in value:

                search_button = button
                break

    if search_button is None:

        raise RuntimeError(
            "Could not find Search button."
        )

    search_button.click()

    page.wait_for_timeout(3000)

    print("✓ Search completed.")
    print(f"Current URL: {page.url}")


# ============================================================
# FIND TRACK TABLE
# ============================================================

def find_track_table(page):

    tables = page.locator("table")

    print()
    print(
        f"Searching {tables.count()} tables "
        f"for the track table..."
    )

    for table_index in range(tables.count()):

        table = tables.nth(table_index)

        rows = table.locator("tr")

        if rows.count() == 0:
            continue

        # Check each row because the first row of the object
        # information table also contains useful text.
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

                print(
                    f"✓ Track table found "
                    f"(table {table_index})."
                )

                return table

    return None


# ============================================================
# EXTRACT TRACK ID
# ============================================================

def extract_track_id(row):

    links = row.locator("a")

    # The first link in a track row is the Track ID.
    for i in range(links.count()):

        link = links.nth(i)

        text = link.inner_text().strip()

        if re.fullmatch(r"\d+", text):

            return text

    # Fallback: find a long integer.
    text = row.inner_text()

    match = re.search(
        r"\b\d{6,}\b",
        text
    )

    if match:

        return match.group(0)

    return None


# ============================================================
# EXTRACT PERIOD
# ============================================================

def extract_period(row):

    text = row.inner_text()

    # Aperiodic tracks are explicitly labelled.
    if re.search(
        r"\bAperiodic\b",
        text,
        re.IGNORECASE
    ):

        return None

    # Periodic tracks have:
    #
    # Period: 192.19 s
    #

    match = re.search(
        r"Period:\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        text,
        re.IGNORECASE
    )

    if match:

        return float(
            match.group(1)
        )

    return None


# ============================================================
# EXTRACT PRODUCT LINKS FROM A TRACK
# ============================================================

def extract_product_links(row):

    links = row.locator("a")

    products = []

    for i in range(links.count()):

        link = links.nth(i)

        label = link.inner_text().strip()

        href = link.get_attribute("href")

        if not label:
            continue

        # Product labels are generally single letters.
        # Ignore the Track ID link.
        if label.upper() in PRODUCT_LABELS:

            products.append(
                {
                    "label": label.upper(),
                    "href": href,
                    "index": i
                }
            )

    return products


# ============================================================
# EXTRACT PERIODIC TRACKS
# ============================================================

def extract_periodic_tracks(table):

    print()
    print("=" * 60)
    print("ANALYSING TRACKS")
    print("=" * 60)

    rows = table.locator("tr")

    tracks = []

    for row_index in range(rows.count()):

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

        # ----------------------------------------------------
        # Aperiodic → skip
        # ----------------------------------------------------

        if period is None:

            print(
                f"{track_id:<12} "
                f"Aperiodic → SKIP"
            )

            continue

        products = extract_product_links(row)

        print(
            f"{track_id:<12} "
            f"Period = {period:8.2f} s   "
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

    print()
    print(
        f"Periodic tracks found: {len(tracks)}"
    )

    return tracks


# ============================================================
# DOWNLOAD ONE PRODUCT USING THE BROWSER
# ============================================================

def download_product(
    page,
    product_link,
    track_dir,
    track_id
):
    """
    Download one MMT-9 product using a real browser click.

    We do NOT use page.request.get(), because that was producing
    HTTP 403 from MMT-9.

    Instead we:
        1. Open the product in a new browser tab.
        2. Let Chromium make the request.
        3. If the result contains an image, save the rendered
           image as PNG.
    """

    label = product_link["label"]
    href = product_link["href"]

    if not href:

        print(
            f"    {label}: no href"
        )

        return None

    # --------------------------------------------------------
    # Open a fresh page within the SAME browser context.
    # This preserves cookies/session information.
    # --------------------------------------------------------

    product_page = page.context.new_page()

    try:

        # We first navigate to the object page in the new tab.
        #
        # This gives the browser the same MMT-9 context before
        # following the product link.
        product_page.goto(
            page.url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        product_page.wait_for_timeout(500)

        # ----------------------------------------------------
        # Locate the exact product link using its href.
        # ----------------------------------------------------

        product_link_locator = product_page.locator(
            f'a[href="{href}"]'
        )

        if product_link_locator.count() == 0:

            # Fallback: use the product label.
            product_link_locator = (
                product_page.get_by_text(
                    label,
                    exact=True
                )
            )

        if product_link_locator.count() == 0:

            print(
                f"    {label}: link not found"
            )

            return None

        link = product_link_locator.first

        # ----------------------------------------------------
        # Force the link to open in the same new tab.
        #
        # This is still a genuine browser navigation, unlike
        # page.request.get().
        # ----------------------------------------------------

        link.evaluate(
            """
            element => {
                element.target = "_self";
            }
            """
        )

        print(
            f"    Opening {label}..."
        )

        link.click()

        product_page.wait_for_load_state(
            "domcontentloaded",
            timeout=60000
        )

        product_page.wait_for_timeout(1500)

        # ----------------------------------------------------
        # Look for an image.
        # ----------------------------------------------------

        images = product_page.locator("img")

        if images.count() == 0:

            print(
                f"    {label}: no image found"
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

        # Save the image rendered by Chromium.
        image.screenshot(
            path=str(destination)
        )

        size_kb = (
            destination.stat().st_size / 1024
        )

        print(
            f"    ✓ {filename} "
            f"({size_kb:.1f} KB)"
        )

        return filename

    except Exception as e:

        print(
            f"    ✗ {label}: {e}"
        )

        return None

    finally:

        product_page.close()


# ============================================================
# DOWNLOAD PRODUCTS FOR ONE TRACK
# ============================================================

def download_track_products(
    page,
    track,
    object_dir
):

    track_id = track["track_id"]

    track_dir = (
        object_dir / track_id
    )

    track_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("-" * 60)
    print(
        f"TRACK {track_id}"
    )
    print(
        f"Period: {track['period']:.2f} s"
    )
    print("-" * 60)

    downloaded = []

    for product in track["products"]:

        filename = download_product(
            page,
            product,
            track_dir,
            track_id
        )

        if filename:

            downloaded.append(filename)

        # Don't hammer the server.
        time.sleep(0.5)

    return downloaded


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    object_dir,
    records
):

    manifest_path = (
        object_dir / "manifest.csv"
    )

    with open(
        manifest_path,
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

        for record in records:

            writer.writerow(record)

    print()
    print(
        f"✓ Manifest saved:"
    )

    print(
        f"  {manifest_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("MMT-9 DATA FETCHER")
    print("=" * 60)
    print()
    print(f"NORAD ID : {NORAD_ID}")
    print(f"MMT-9    : {MMT9_URL}")
    print(f"Output   : {DATA_DIR}")
    print()

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

            # ------------------------------------------------
            # STEP 1 — Open MMT-9
            # ------------------------------------------------

            if not open_mmt9(page):

                raise RuntimeError(
                    "Could not load MMT-9."
                )

            # ------------------------------------------------
            # STEP 2 — Search NORAD
            # ------------------------------------------------

            search_norad(
                page,
                NORAD_ID
            )

            # ------------------------------------------------
            # STEP 3 — Find track table
            # ------------------------------------------------

            table = find_track_table(page)

            if table is None:

                raise RuntimeError(
                    "Could not find the track table."
                )

            # ------------------------------------------------
            # STEP 4 — Extract periodic tracks
            # ------------------------------------------------

            tracks = (
                extract_periodic_tracks(table)
            )

            if not tracks:

                print(
                    "No periodic tracks found."
                )

                return

            # ------------------------------------------------
            # STEP 5 — Download
            # ------------------------------------------------

            print()
            print("=" * 60)
            print("DOWNLOADING PERIODIC TRACKS")
            print("=" * 60)

            manifest_records = []

            for number, track in enumerate(
                tracks,
                start=1
            ):

                print()
                print(
                    f"[{number}/{len(tracks)}]"
                )

                downloaded = (
                    download_track_products(
                        page,
                        track,
                        DATA_DIR
                    )
                )

                manifest_records.append(
                    {
                        "norad_id": NORAD_ID,
                        "track_id":
                            track["track_id"],
                        "period_sec":
                            track["period"],
                        "downloaded_products":
                            ";".join(downloaded)
                    }
                )

            # ------------------------------------------------
            # STEP 6 — Save manifest
            # ------------------------------------------------

            save_manifest(
                DATA_DIR,
                manifest_records
            )

            # ------------------------------------------------
            # Finished
            # ------------------------------------------------

            print()
            print("=" * 60)
            print("FETCH COMPLETE")
            print("=" * 60)
            print()
            print(
                f"Periodic tracks: {len(tracks)}"
            )
            print(
                f"Output: {DATA_DIR}"
            )

            input(
                "\nPress ENTER to close..."
            )

        except Exception as e:

            print()
            print("=" * 60)
            print("ERROR")
            print("=" * 60)
            print()
            print(e)

            input(
                "\nPress ENTER to close..."
            )

        finally:

            context.close()
            browser.close()


if __name__ == "__main__":
    main()