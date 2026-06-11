#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile


def _parse_case(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each --case must have the form name=/path/to/file.h5")

    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("Case name must not be empty.")
    if not path:
        raise argparse.ArgumentTypeError("Case path must not be empty.")
    return name, path


def _run(command, env):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run local VLX -> TREXIO regression checks.")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
        help="Regression case in the form name=/absolute/or/relative/path/to/file.h5",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the converter and comparison utility",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where converted TREXIO files are written; defaults to a temporary directory",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep converted TREXIO files when using a temporary output directory",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_root = os.path.join(repo_root, "src")
    compare_script = os.path.join(repo_root, "utilities", "compare_vlx_trexio.py")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not existing_pythonpath else source_root + os.pathsep + existing_pythonpath

    temporary_directory = None
    output_dir = args.output_dir
    if output_dir is None:
        if args.keep_output:
            output_dir = tempfile.mkdtemp(prefix="trexio-vlx-regression-")
        else:
            temporary_directory = tempfile.TemporaryDirectory(prefix="trexio-vlx-regression-")
            output_dir = temporary_directory.name
    else:
        os.makedirs(output_dir, exist_ok=True)

    try:
        for name, case_path in args.case:
            input_path = os.path.abspath(case_path)
            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"VLX input file not found: {input_path}")

            output_path = os.path.join(output_dir, f"{name}.hdf5")

            print(f"=== {name} ===", flush=True)
            _run(
                [
                    args.python,
                    "-m",
                    "trexio_tools.trexio_run",
                    "convert-from",
                    "-w",
                    "-t",
                    "vlx",
                    "-i",
                    input_path,
                    "-b",
                    "hdf5",
                    output_path,
                ],
                env,
            )
            _run([args.python, compare_script, input_path, output_path], env)

        print("All VLX regression cases passed.", flush=True)
        return 0
    finally:
        if temporary_directory is not None and not args.keep_output:
            temporary_directory.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())