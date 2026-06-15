from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
import pandas as pd


def make_training_archive(output_dir: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    print("Creating archive:")
    print(archive_path)

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)

    print("Archive created:")
    print(archive_path)
    return archive_path

def extract_training_archives(
    archives_dir: str | Path,
    archive_names: list[str],
    extract_dir: str | Path | None = None,
):
    archives_dir = Path(archives_dir)

    if extract_dir is None:
        extract_dir = Path(os.environ.get("TMPDIR", "/tmp"))

    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_dirs = []

    for archive_name in archive_names:

        archive_path = archives_dir / archive_name

        if not archive_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono archiwum: {archive_path}"
            )

        if archive_name.endswith(".tar.gz"):
            experiment_name = archive_name[:-7]
        elif archive_name.endswith(".tgz"):
            experiment_name = archive_name[:-4]
        else:
            experiment_name = Path(archive_name).stem

        expected_dir = extract_dir / experiment_name

        if expected_dir.exists():
            print(f"[SKIP] {experiment_name} already extracted")
            extracted_dirs.append(expected_dir)
            continue

        print(f"[EXTRACT] {archive_name}")

        if archive_name.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(extract_dir)

        elif archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        else:
            raise ValueError(
                f"Nieobsługiwany format archiwum: {archive_name}"
            )

        if not expected_dir.exists():
            raise FileNotFoundError(
                f"Po rozpakowaniu nie znaleziono katalogu eksperymentu: {expected_dir}"
            )

        extracted_dirs.append(expected_dir)

    return extracted_dirs

def prepare_dataset_from_tar(dataset_tar: Path, data_dir: Path, skip_if_exists: bool = True) -> Path:
    if not dataset_tar.exists():
        raise FileNotFoundError(f"Nie znaleziono archiwum: {dataset_tar}")

    if data_dir.exists() and skip_if_exists:
        print(f"Dataset już istnieje: {data_dir}")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rozpakowywanie: {dataset_tar}")
    print(f"Do: {data_dir}")

    with tarfile.open(dataset_tar, "r") as tar:
        tar.extractall(data_dir)


    return data_dir