import argparse
import os
from pathlib import Path

import numpy as np
import orjson
import soundfile as sf
from data.semantic_dataset import TextTokenizer
from torch.utils.data import Dataset
from torchaudio.datasets import LJSPEECH
from tqdm import tqdm


def read_jsonl(f_path):
    data = []
    with open(f_path, "rb") as f:
        for line in f:
            data.append(orjson.loads(line))
    return data


class LADA(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self._path = self.data_root / "accept"
        self.manifest_path = self._path / "metadata.jsonl"
        self._flist = self.process_manifest()

    def process_manifest(self):
        raw_manifest = read_jsonl(self.manifest_path)
        out_manifest = []
        for itm in raw_manifest:
            file_id, _ = os.path.splitext(itm["file"])
            out_manifest.append([file_id, itm["orig_text"], itm["orig_text_wo_stress"]])
        return out_manifest

    def __len__(self):
        return len(self._flist)


def load_dataset(args):
    data_root = args.data_root
    if args.dataset == "ljspeech":
        dataset = LJSPEECH(data_root, download=True)
    elif args.dataset == "lada":
        dataset = LADA(data_root)
    else:
        raise ValueError
    return dataset


def split_and_write_manifests(dataset, args):
    data_root = args.data_root
    dataset_idxs = np.arange(start=0, stop=len(dataset))
    np.random.shuffle(dataset_idxs)
    test_idxs, val_idxs, train_idxs = (
        dataset_idxs[:300],
        dataset_idxs[300:600],
        dataset_idxs[600:],
    )

    print(f"{len(test_idxs)=}")
    print(f"{len(val_idxs)=}")
    print(f"{len(train_idxs)=}")
    dataset_items = dataset._flist
    test_data, val_data, train_data = dict(), dict(), dict()
    phonemizer = TextTokenizer()
    for idx, itm in tqdm(enumerate(dataset_items)):
        file_id, raw_text, text = itm
        file_id = file_id + ".wav"
        wav_path = dataset._path / file_id
        wav_obj = sf.SoundFile(wav_path)
        duration = wav_obj.frames / wav_obj.samplerate

        phones = phonemizer(text)[0]
        phones = "|".join(phones)

        datapoint = {
            file_id: {
                "text": text,
                "raw-text": raw_text,
                "duration": duration,
                "phoneme": phones,
            }
        }
        if idx in test_idxs:
            test_data.update(datapoint)
        elif idx in val_idxs:
            val_data.update(datapoint)
        elif idx in train_idxs:
            train_data.update(datapoint)

    test_manifest_path = data_root / "test.json"
    val_manifest_path = data_root / "dev.json"
    train_manifest_path = data_root / "train.json"

    with open(test_manifest_path, "wb") as f:
        f.write(orjson.dumps(test_data, option=orjson.OPT_INDENT_2))

    with open(val_manifest_path, "wb") as f:
        f.write(orjson.dumps(val_data, option=orjson.OPT_INDENT_2))

    with open(train_manifest_path, "wb") as f:
        f.write(orjson.dumps(train_data, option=orjson.OPT_INDENT_2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ljspeech")
    # fmt: off
    parser.add_argument("--data_root", type=Path, default="./datasets/ljspeech-training-data")
    # fmt: on
    args = parser.parse_args()
    args.data_root.mkdir(exist_ok=True)

    np.random.seed(42)
    dataset = load_dataset(args)
    split_and_write_manifests(dataset, args)


if __name__ == "__main__":
    main()
