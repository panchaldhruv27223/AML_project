import os
import csv
import torchaudio
import logging
from tqdm import tqdm
import textgrid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_csv(audio_dir, annotation_dir, textgrid_dir, csv_file, duration_limits=(0.3, 20.0)):
    """Create CSV with audio, annotations, and timestamps."""
    logger.info(f"Creating CSV: {csv_file}")
    audio_files = [os.path.join(root, f) for root, _, files in os.walk(audio_dir) for f in files if f.endswith(".flac")]
    audio_files.sort()
    csv_lines = [["ID", "duration", "wav", "wrd", "pos", "gov", "dep", "start_word", "end_word"]]
    total_duration = 0.0

    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        utterance_id = os.path.basename(audio_path).replace(".flac", "")
        rel_path = os.path.relpath(audio_path, audio_dir)
        annotation_path = os.path.join(annotation_dir, os.path.dirname(rel_path), f"{utterance_id}.dep")
        textgrid_path = os.path.join(textgrid_dir, os.path.dirname(rel_path), f"{utterance_id}.TextGrid")

        # if not all(os.path.isfile(p) for p in [audio_path, annotation_path, textgrid_path]):
        #     logger.warning(f"Missing files for {utterance_id}")
        #     continue

        # Check duration
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        if duration < duration_limits[0] or duration > duration_limits[1]:
            logger.warning(f"Duration {duration}s out of bounds for {utterance_id}")
            continue
        total_duration += duration

        # Read annotations
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        words = [line.split('\t')[1] for line in lines]
        pos = [line.split('\t')[2] for line in lines]
        heads = [line.split('\t')[3] for line in lines]
        deprels = [line.split('\t')[4] for line in lines]

        # Read timestamps
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        timestamps = []
        for tier in tg.tiers:
            if tier.name.lower() == 'words':
                timestamps = [(interval.mark, interval.minTime, interval.maxTime) for interval in tier.intervals if interval.mark]
                break

        if len(timestamps) != len(words):
            logger.warning(f"Word count mismatch for {utterance_id}: {len(timestamps)} timestamps vs {len(words)} words")
            continue

        start_words = [str(t[1]) for t in timestamps]
        end_words = [str(t[2]) for t in timestamps]

        csv_line = [
            utterance_id,
            str(duration),
            audio_path,
            " ".join(words).upper(),
            " ".join(pos),
            " ".join(heads),
            " ".join(deprels),
            " ".join(start_words),
            " ".join(end_words)
        ]
        csv_lines.append(csv_line)

    with open(csv_file, mode="w", encoding="utf-8", newline='') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in csv_lines:
            csv_writer.writerow(line)
    logger.info(f"{csv_file} created! Samples: {len(csv_lines)-1}, Total duration: {round(total_duration/3600, 2)} hours")

def prepare_librispeech(
    data_folder,
    save_folder,
    train_folder="train-clean-360",
    dev_folder="dev-clean",
    test_folder="test-clean",
    skip_prep=False
):
    """Prepare CSV files for LibriSpeech."""
    if skip_prep:
        logger.info("Skipping preparation")
        return
    os.makedirs(save_folder, exist_ok=True)
    save_csv_train = os.path.join(save_folder, "train.csv")
    save_csv_dev = os.path.join(save_folder, "dev.csv")
    save_csv_test = os.path.join(save_folder, "test.csv")

    if all(os.path.isfile(f) for f in [save_csv_train, save_csv_dev, save_csv_test]):
        logger.info("CSV files already exist, skipping preparation!")
        return

    for subset, save_csv in [
        (train_folder, save_csv_train),
        (dev_folder, save_csv_dev),
        (test_folder, save_csv_test)
    ]:
        if subset:
            create_csv(
                os.path.join(data_folder, subset),
                os.path.join(data_folder, f"{subset}-annotated"),
                os.path.join(data_folder, subset),
                save_csv
            )

if __name__ == "__main__":
    prepare_librispeech(
        data_folder=r"D:\Manish Prajapati\LibriSpeech",
        save_folder=r"D:\Manish Prajapati\LibriSpeech\csv",
        train_folder="train-clean-360",
        dev_folder="dev-clean",
        test_folder="test-clean"
    )