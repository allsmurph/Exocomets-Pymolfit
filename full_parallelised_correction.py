import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pymolfit
from pymolfit import Spectrum, correct, select_telluric_regions, AdvancedOptions
#from plotting_params import use_my_style
import time
#use_my_style()
from joblib import Parallel, delayed
pymolfit_version = pymolfit.__version__
print(pymolfit_version)

start_time = time.time()

base_dir = Path("/srv/local/anba6163/Clean_Data/")

fits_files = sorted(
    file for file in base_dir.rglob("cr2res_obs_nodding_extracted_combined.fits")
    if "old" not in file.parts)

observations = []
for file in fits_files:
    date_dir = file.parent.parent.name

    if "M" not in date_dir:
        continue

    region_dir = (
        Path("/home/alst3712/exocomet_project/clean_data")
        / date_dir
        / "order_regions"
    )
    region_dir.mkdir(parents=True, exist_ok=True)

    observations.append((file, region_dir))

print(f"Found {len(observations)} M-band observations")


def get_order_tasks(FILE, DIR, start_pixel, end_pixel):
    """Create one task per selected FITS order."""
    tasks = []

    with fits.open(FILE) as hdus:
        for chip, hdu in enumerate(hdus[1:], 1):
            wl_columns = [
                name for name in hdu.columns.names
                if name.endswith("_WL")
            ]

            # Preserve your current behaviour: first two orders per chip.
            for order_index, wl in enumerate(wl_columns[:]):
                order = wl.removesuffix("_WL")
                spec = f"{order}_SPEC"
                err = f"{order}_ERR"

                # Only schedule orders whose matching columns exist.
                if spec not in hdu.columns.names or err not in hdu.columns.names:
                    print(
                        f"Skipping chip {chip}, order {order}: "
                        "missing flux or uncertainty column"
                    )
                    continue

                tasks.append((
                    FILE, DIR, chip, wl, spec, err,
                    order_index, start_pixel, end_pixel
                ))

    return tasks


def process_order(
    FILE, DIR, chip, wl, spec, err,
    order_index, start_pixel, end_pixel
):
    """Correct a single order in a worker process."""
    date_dir = FILE.parent.parent.name
    order = wl.removesuffix("_WL")

    try:
        print(
            f"Working on {date_dir}, chip {chip}, order {order}",
            flush=True,
        )

        result = correct(
            input_path=FILE,
            hdu=chip,
            wavelength_col=wl,
            flux_col=spec,
            uncertainty_col=err,
            wavelength_medium="vacuum",
            wavelength_unit="nm",
        )

        corrected = result.corrected.to_unit("nm")

        data = np.column_stack((
            corrected.wavelength[start_pixel:end_pixel],
            corrected.flux[start_pixel:end_pixel],
            corrected.uncertainty[start_pixel:end_pixel],
        ))

        return {
            "file": str(FILE),
            "dir": str(DIR),
            "chip": chip,
            "order_index": order_index,
            "order": order,
            "data": data,
            "error": None,
        }

    except Exception as exc:
        return {
            "file": str(FILE),
            "dir": str(DIR),
            "chip": chip,
            "order_index": order_index,
            "order": order,
            "data": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def save_observation(results, FILE, DIR):
    """Combine successful orders and write one file per observation."""
    output_file = (
        DIR /
        f"tellurics_corrected_version_{pymolfit_version}_"
        f"{FILE.parent.parent.name}_base.dat"
    )
    temp_file = output_file.with_suffix(".dat.tmp")

    # Keep the original chip/order traversal order.
    results = sorted(
        results,
        key=lambda r: (r["chip"], r["order_index"]),
    )

    corrected_orders = [
        r["data"] for r in results
        if r["data"] is not None
    ]

    if not corrected_orders:
        print(
            f"No successful orders for {FILE.parent.parent.name}. "
            "No output written."
        )
        return

    corrected_data = np.vstack(corrected_orders)

    try:
        with open(temp_file, "w") as f:
            f.write("# wavelength flux error\n")
            np.savetxt(
                f,
                corrected_data,
                fmt=["%.10f", "%.10e", "%.10e"],
            )

        os.replace(temp_file, output_file)

    finally:
        if temp_file.exists():
            temp_file.unlink()

    failed = [r for r in results if r["error"] is not None]

    print(
        f"Saved {output_file} "
        f"({len(corrected_orders)} successful orders, "
        f"{len(failed)} failed)"
    )


# %%
if __name__ == "__main__":
    start_pixel = 40
    end_pixel = -40

    if observations:
        # Build a single pool of order-level tasks across all observations.
        all_tasks = []

        for FILE, DIR in observations[:]:
            all_tasks.extend(
                get_order_tasks(FILE, DIR, start_pixel, end_pixel)
            )

        print(f"Total order tasks: {len(all_tasks)}")
        n_jobs = 50

        # Parallelise across orders, rather than across entire FITS files.
        results = Parallel(
            n_jobs=n_jobs,
            verbose=10,
        )(
            delayed(process_order)(*task)
            for task in all_tasks
        )

        # Report failures in the main process.
        for r in results:
            if r["error"] is not None:
                print(
                    f"FAILED: file={Path(r['file']).parent.parent.name}, "
                    f"chip={r['chip']}, order={r['order']}, "
                    f"error={r['error']}",
                    flush=True,
                )

        # Group results by observation and write each output once.
        for FILE, DIR in observations[:]:
            file_results = [
                r for r in results
                if r["file"] == str(FILE)
            ]
            save_observation(file_results, FILE, DIR)

    print(
        f"\nTotal time taken: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

