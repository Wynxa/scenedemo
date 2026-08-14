from __future__ import annotations

import array
import os
import sys
import zipfile
from typing import Any

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm

from services.scene_graph_service.app.structures.boxlist_lite import BoxListLite


SEMANTIC_ALIASES = {
    "__background__": [],
    "bricklaying_zone": ["bricklaying area", "bricklaying", "area"],
    "face_mask": ["face mask", "mask"],
    "rebar_zone": ["rebar area", "rebar", "area"],
    "unprotected_edge": ["unprotected edge", "edge"],
    "co_working_with": ["working with", "work with"],
    "next_to": ["next to", "next"],
    "standing_on": ["standing on", "standing"],
}


def cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def to_onehot(vec: torch.Tensor, num_classes: int, fill: float = 1000.0) -> torch.Tensor:
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)
    onehot_result.view(-1)[vec.long() + num_classes * arange_inds] = fill
    return onehot_result


def encode_box_info(proposals: list[BoxListLite]) -> torch.Tensor:
    if not proposals:
        return torch.zeros((0, 9), dtype=torch.float32)
    assert proposals[0].mode == "xyxy"
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        wid, hei = proposal.size
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        if wid * hei == 0:
            raise ValueError("Invalid image size: {} x {}".format(wid, hei))
        info = torch.cat(
            [
                w / wid,
                h / hei,
                x / wid,
                y / hei,
                x1 / wid,
                y1 / hei,
                x2 / wid,
                y2 / hei,
                w * h / (wid * hei),
            ],
            dim=-1,
        ).view(-1, 9)
        boxes_info.append(info)
    return torch.cat(boxes_info, dim=0)


def layer_init(layer: torch.nn.Module, init_para: float = 0.1, normal: bool = False, xavier: bool = True) -> None:
    xavier = False if normal else xavier
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    if xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)


def fusion_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y) - (x - y) ** 2


def _candidate_phrases(token: str) -> list[str]:
    alias_list = SEMANTIC_ALIASES.get(token, [])
    normalized = token.replace("_", " ")
    candidates = []
    if token != "__background__":
        candidates.append(token)
        if normalized != token:
            candidates.append(normalized)
    candidates.extend(alias_list)

    unique: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        unique.append(cand)
        seen.add(cand)
    return unique


def _lookup_phrase_vector(phrase: str, wv_dict: dict[str, int], wv_arr: torch.Tensor, wv_dim: int) -> torch.Tensor | None:
    wv_index = wv_dict.get(phrase, None)
    if wv_index is not None:
        return wv_arr[wv_index]

    split_tokens = phrase.split(" ")
    pieces = []
    for piece in split_tokens:
        wv_index = wv_dict.get(piece, None)
        if wv_index is not None:
            pieces.append(wv_arr[wv_index])
    if pieces:
        return torch.stack(pieces, dim=0).mean(dim=0)
    return None


def rel_vectors(names: list[str], wv_dir: str, wv_type: str = "glove.6B", wv_dim: int = 300) -> torch.Tensor:
    wv_dict, wv_arr, _ = load_word_vectors(wv_dir, wv_type, wv_dim)
    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)
    for i, token in enumerate(names):
        if i == 0:
            continue
        found = None
        for phrase in _candidate_phrases(token):
            found = _lookup_phrase_vector(phrase, wv_dict, wv_arr, wv_dim)
            if found is not None:
                vectors[i] = found
                break
        if found is None:
            print("fail on {}".format(token))
    return vectors


def obj_edge_vectors(names: list[str], wv_dir: str, wv_type: str = "glove.6B", wv_dim: int = 300) -> torch.Tensor:
    wv_dict, wv_arr, _ = load_word_vectors(wv_dir, wv_type, wv_dim)
    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)
    for i, token in enumerate(names):
        if token == "__background__":
            vectors[i].zero_()
            continue
        found = None
        for phrase in _candidate_phrases(token):
            found = _lookup_phrase_vector(phrase, wv_dict, wv_arr, wv_dim)
            if found is not None:
                vectors[i] = found
                break
        if found is None:
            print("fail on {}".format(token))
    return vectors


def load_word_vectors(root: str, wv_type: str, dim: int | str) -> tuple[dict[str, int], torch.Tensor, int]:
    url_map = {
        "glove.42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
        "glove.840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "glove.twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "glove.6B": "http://nlp.stanford.edu/data/glove.6B.zip",
    }
    if isinstance(dim, int):
        dim = str(dim) + "d"
    fname = os.path.join(root, wv_type + "." + dim)

    if os.path.isfile(fname + ".pt"):
        fname_pt = fname + ".pt"
        print("loading word vectors from", fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as exc:
            print("Error loading the model from {}{}".format(fname_pt, str(exc)))
            sys.exit(-1)

    if os.path.isfile(fname + ".txt"):
        fname_txt = fname + ".txt"
        content = [line for line in open(fname_txt, "rb")]
    elif os.path.basename(wv_type) in url_map:
        url = url_map[wv_type]
        print("downloading word vectors from {}".format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit="B", unit_scale=True, miniters=1, desc=filename) as progress:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(progress))
            with zipfile.ZipFile(fname, "r") as zip_ref:
                print("extracting word vectors into {}".format(root))
                zip_ref.extractall(root)
        if not os.path.isfile(fname + ".txt"):
            raise RuntimeError("no word vectors of requested dimension found")
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError("unable to load word vectors from {}".format(root))

    wv_tokens: list[str] = []
    wv_arr = array.array("d")
    wv_size: int | None = None
    for line in tqdm(range(len(content)), desc="loading word vectors from {}".format(fname_txt)):
        entries = content[line].strip().split(b" ")
        word, entries = entries[0], entries[1:]
        if wv_size is None:
            wv_size = len(entries)
        try:
            if isinstance(word, six.binary_type):
                word = word.decode("utf-8")
        except Exception:
            print("non-UTF8 token", repr(word), "ignored")
            continue
        wv_arr.extend(float(x) for x in entries)
        wv_tokens.append(word)

    if wv_size is None:
        raise RuntimeError("word vectors file is empty: {}".format(fname_txt))
    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_tensor = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_tensor, wv_size)
    torch.save(ret, fname + ".pt")
    return ret


def reporthook(progress: Any):
    last_b = [0]

    def inner(b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            progress.total = tsize
        progress.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
