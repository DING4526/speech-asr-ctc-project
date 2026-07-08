from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


def read_pinyin_file(path: str | Path) -> Dict[str, List[str]]:
    """Read a pinyin manifest.

    Expected format per line:
        utt_id<TAB>pin yin token sequence
    Spaces between pinyin tokens are required.
    """
    path = Path(path)
    items: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{lineno} should contain a TAB between utt_id and text")
            utt_id, text = line.split("\t", 1)
            tokens = text.strip().split()
            if not tokens:
                raise ValueError(f"{path}:{lineno} has empty pinyin sequence")
            items[utt_id] = tokens
    return items


def build_vocab_from_files(text_files: Iterable[str | Path], vocab_path: str | Path) -> None:
    """Build vocabulary from pinyin files.

    Index 0 is reserved for CTC blank. Index 1 is reserved for unknown tokens.
    """
    token_set = set()
    for text_file in text_files:
        if not Path(text_file).exists():
            continue
        for tokens in read_pinyin_file(text_file).values():
            token_set.update(tokens)

    if not token_set:
        raise RuntimeError("No pinyin tokens found. Please check your pinyin files.")

    vocab = [BLANK_TOKEN, UNK_TOKEN] + sorted(token_set)
    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def load_vocab(vocab_path: str | Path) -> tuple[Dict[str, int], Dict[int, str]]:
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    tokens = [line.rstrip("\n") for line in path.open("r", encoding="utf-8") if line.strip()]
    token2id = {tok: i for i, tok in enumerate(tokens)}
    id2token = {i: tok for tok, i in token2id.items()}
    if token2id.get(BLANK_TOKEN, None) != 0:
        raise ValueError("Vocabulary must put <blank> at index 0 for CTC.")
    return token2id, id2token


def encode_tokens(tokens: List[str], token2id: Dict[str, int]) -> List[int]:
    unk_id = token2id[UNK_TOKEN]
    return [token2id.get(tok, unk_id) for tok in tokens]


def decode_ids(ids: List[int], id2token: Dict[int, str], remove_special: bool = True) -> List[str]:
    tokens = []
    for idx in ids:
        token = id2token.get(int(idx), UNK_TOKEN)
        if remove_special and token in {BLANK_TOKEN, UNK_TOKEN}:
            continue
        tokens.append(token)
    return tokens
