import argparse
import random
import shutil
import re
from pathlib import Path

RANDOM_SEED = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_IMAGES_PER_SPLIT = 3

random.seed(RANDOM_SEED)


def extract_trajectory_id(image_path):
    match = re.match(r"fish_([0-9]+)_([0-9]+)\.(png|jpg|jpeg)$", image_path.name)

    if not match:
        raise ValueError(f"Niepoprawna nazwa pliku: {image_path.name}")

    return match.group(1)


def group_images_by_trajectory(image_paths):
    trajectories = {}

    for image_path in image_paths:
        trajectory_id = extract_trajectory_id(image_path)

        if trajectory_id not in trajectories:
            trajectories[trajectory_id] = []

        trajectories[trajectory_id].append(image_path)

    return trajectories


def count_images(trajectory_ids, trajectories):
    return sum(len(trajectories[tid]) for tid in trajectory_ids)


def split_class_trajectories(image_paths):
    image_paths = list(image_paths)

    if len(image_paths) < 9:
        raise ValueError(
            f"Klasa posiada tylko {len(image_paths)} obrazów. "
            f"Minimum to 9."
        )

    trajectories = group_images_by_trajectory(image_paths)

    trajectory_ids = list(trajectories.keys())

    if len(trajectory_ids) < 3:
        raise ValueError(
            f"Klasa posiada tylko {len(trajectory_ids)} trajektorie. "
            f"Minimum to 3."
        )

    random.shuffle(trajectory_ids)

    total_images = len(image_paths)

    target_val = max(MIN_IMAGES_PER_SPLIT, round(total_images * VAL_RATIO))
    target_test = max(MIN_IMAGES_PER_SPLIT, round(total_images * TEST_RATIO))

    val_ids = []
    test_ids = []
    train_ids = []

    sorted_ids = trajectory_ids[:]
    random.shuffle(sorted_ids)

    for tid in sorted_ids:
        if count_images(val_ids, trajectories) < target_val:
            val_ids.append(tid)
        elif count_images(test_ids, trajectories) < target_test:
            test_ids.append(tid)
        else:
            train_ids.append(tid)

    if count_images(val_ids, trajectories) < MIN_IMAGES_PER_SPLIT:
        raise ValueError("Nie udało się uzyskać minimum obrazów w val.")

    if count_images(test_ids, trajectories) < MIN_IMAGES_PER_SPLIT:
        raise ValueError("Nie udało się uzyskać minimum obrazów w test.")

    if count_images(train_ids, trajectories) < MIN_IMAGES_PER_SPLIT:
        raise ValueError("Nie udało się uzyskać minimum obrazów w train.")

    train = []
    val = []
    test = []

    for tid in train_ids:
        train.extend(trajectories[tid])

    for tid in val_ids:
        val.extend(trajectories[tid])

    for tid in test_ids:
        test.extend(trajectories[tid])

    return train, val, test, train_ids, val_ids, test_ids


def copy_images(images, output_dir, split_name, class_name):
    target_dir = output_dir / split_name / class_name

    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        target_path = target_dir / image_path.name
        shutil.copy2(image_path, target_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_dir",
        required=True,
        help="Folder źródłowy zawierający fish_01, fish_02, ..."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder wyjściowy"
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        print(f"[INFO] Usuwanie istniejącego folderu: {output_dir}")
        shutil.rmtree(output_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Nie znaleziono source_dir: {source_dir}"
        )

    class_dirs = sorted(
        [p for p in source_dir.iterdir() if p.is_dir()]
    )

    if not class_dirs:
        raise ValueError(
            f"Nie znaleziono folderów klas w: {source_dir}"
        )

    summary = []

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_paths = sorted(
            list(class_dir.glob("*.png")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg"))
        )

        if not image_paths:
            print(f"[WARNING] Brak zdjęć w klasie: {class_name}")
            continue

        try:
            train, val, test, train_ids, val_ids, test_ids = split_class_trajectories(image_paths)

        except ValueError as e:
            print(f"[WARNING] {class_name}: {e}")
            continue

        copy_images(train, output_dir, "train", class_name)
        copy_images(val, output_dir, "val", class_name)
        copy_images(test, output_dir, "test", class_name)

        total = len(image_paths)

        summary.append(
            (
                class_name,
                total,
                len(train),
                len(val),
                len(test),
                len(train_ids),
                len(val_ids),
                len(test_ids)
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "split_report.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "class_name,total_images,"
            "train_images,val_images,test_images,"
            "train_percent,val_percent,test_percent,"
            "train_trajectories,val_trajectories,test_trajectories\n"
        )

        for row in summary:
            total = row[1]

            train_percent = row[2] / total * 100
            val_percent = row[3] / total * 100
            test_percent = row[4] / total * 100

            f.write(
                f"{row[0]},{total},"
                f"{row[2]},{row[3]},{row[4]},"
                f"{train_percent:.2f},"
                f"{val_percent:.2f},"
                f"{test_percent:.2f},"
                f"{row[5]},{row[6]},{row[7]}\n"
            )

    print("\n========== PODSUMOWANIE ==========\n")
    print("class,total,train,val,test,train%,val%,test%,train_traj,val_traj,test_traj")

    for row in summary:
        total = row[1]

        train_percent = row[2] / total * 100
        val_percent = row[3] / total * 100
        test_percent = row[4] / total * 100

        print(
            f"{row[0]},{total},"
            f"{row[2]},{row[3]},{row[4]},"
            f"{train_percent:.2f}%,"
            f"{val_percent:.2f}%,"
            f"{test_percent:.2f}%,"
            f"{row[5]},{row[6]},{row[7]}"
        )

    print(f"\nRaport zapisano do: {report_path}")


if __name__ == "__main__":
    main()
