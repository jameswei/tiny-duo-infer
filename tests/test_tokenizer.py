"""
Tests for tiny_duo_infer.tokenizer.loader.Tokenizer.

Unit tests use a minimal synthetic tokenizer written to tmp_path — no real
model artifacts required. The fixture creates a small WordLevel tokenizer with:
  - four content tokens (hello=0, world=1, foo=2, bar=3)
  - two tokens registered via add_special_tokens(special=True):
    [BOS]=4, [EOS]=5 — these are filtered by skip_special_tokens=True in decode

Using add_special_tokens() for [BOS]/[EOS] (rather than including them in the
WordLevel vocab dict) is the correct way to make the tokenizers library treat
them as specials. This mirrors how Llama's tokenizer.json registers its
<|begin_of_text|> and <|end_of_text|> tokens.

Slow tests (marked @pytest.mark.slow) require local Llama-3.2-1B artifacts
and are skipped unless --run-slow is passed.
"""

from __future__ import annotations

import json

import pytest
from tokenizers import AddedToken
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from tiny_duo_infer.tokenizer.loader import Tokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Content tokens — regular vocabulary entries.
CONTENT_VOCAB = {"hello": 0, "world": 1, "foo": 2, "bar": 3}

# Special token IDs assigned by add_special_tokens() below.
# add_special_tokens() appends new tokens starting at the current vocab size,
# so [BOS] gets ID 4 and [EOS] gets ID 5 (= len(CONTENT_VOCAB) + 0 and +1).
BOS_ID = 4
EOS_ID = 5
VOCAB_SIZE = 6  # 4 content + 2 special


def _make_hf_tokenizer() -> HFTokenizer:
    """
    Build a minimal WordLevel HF tokenizer with proper special token handling.

    [BOS] and [EOS] are registered via add_special_tokens(special=True) so that
    the tokenizers library tracks them as specials and filters them when
    skip_special_tokens=True is passed to decode().
    """
    tok = HFTokenizer(WordLevel(vocab=CONTENT_VOCAB, unk_token="hello"))
    tok.pre_tokenizer = Whitespace()
    # Register as proper AddedTokens with special=True.
    # These get IDs len(CONTENT_VOCAB)+0 and +1 (i.e. 4 and 5).
    tok.add_special_tokens([
        AddedToken("[BOS]", special=True),
        AddedToken("[EOS]", special=True),
    ])
    # TemplateProcessing prepends [BOS] when add_special_tokens=True,
    # mirroring the TemplateProcessing post-processor in the Llama tokenizer.json.
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A",
        pair="[BOS] $A $B",
        special_tokens=[("[BOS]", BOS_ID)],
    )
    return tok


def _make_hf_tokenizer_without_post_processor() -> HFTokenizer:
    """
    Build a tokenizer with special tokens but no automatic BOS post-processor.

    Qwen3-0.6B advertises add_bos_token=false, so plain prompt-to-completion
    encoding should not assume add_special_tokens=True prepends BOS.
    """
    tok = HFTokenizer(WordLevel(vocab=CONTENT_VOCAB, unk_token="hello"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens([
        AddedToken("[BOS]", special=True),
        AddedToken("[EOS]", special=True),
    ])
    return tok


@pytest.fixture
def tok_path_int_config(tmp_path):
    """
    Synthetic tokenizer with bos_token_id / eos_token_id as direct integers.
    Tests Case 1 of _read_special_token_ids.
    """
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token_id": BOS_ID, "eos_token_id": EOS_ID}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def tok_path_str_config(tmp_path):
    """
    Synthetic tokenizer with bos_token / eos_token as plain strings.
    Tests Case 2 of _read_special_token_ids (string → vocab lookup).
    """
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token": "[BOS]", "eos_token": "[EOS]"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def tok_path_dict_config(tmp_path):
    """
    Synthetic tokenizer with bos_token / eos_token as AddedToken dicts.
    Tests Case 2 with the HF AddedToken dict format used by Llama checkpoints.
    """
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {
        "bos_token": {"content": "[BOS]", "single_word": False, "lstrip": False},
        "eos_token": {"content": "[EOS]", "single_word": False, "lstrip": False},
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    return tmp_path


@pytest.fixture
def tok_path_qwen3_style_config(tmp_path):
    """
    Synthetic Qwen3-style metadata.

    tokenizer_config.json has bos_token=null and eos_token as a string;
    config.json carries bos_token_id/eos_token_id. There is no tokenizer
    post-processor that prepends BOS.
    """
    _make_hf_tokenizer_without_post_processor().save(str(tmp_path / "tokenizer.json"))
    tokenizer_config = {
        "add_bos_token": False,
        "bos_token": None,
        "eos_token": "[EOS]",
    }
    model_config = {
        "bos_token_id": BOS_ID,
        "eos_token_id": EOS_ID,
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    return tmp_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_from_pretrained_int_config(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    assert isinstance(tok, Tokenizer)


def test_from_pretrained_str_config(tok_path_str_config):
    tok = Tokenizer.from_pretrained(tok_path_str_config)
    assert isinstance(tok, Tokenizer)


def test_from_pretrained_dict_config(tok_path_dict_config):
    tok = Tokenizer.from_pretrained(tok_path_dict_config)
    assert isinstance(tok, Tokenizer)


def test_from_pretrained_qwen3_style_config(tok_path_qwen3_style_config):
    tok = Tokenizer.from_pretrained(tok_path_qwen3_style_config)
    assert isinstance(tok, Tokenizer)


def test_missing_tokenizer_json_raises(tmp_path):
    """No tokenizer.json → FileNotFoundError with a clear message."""
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        Tokenizer.from_pretrained(tmp_path)


def test_missing_tokenizer_config_raises(tmp_path):
    """tokenizer.json present but no tokenizer_config.json → FileNotFoundError."""
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    with pytest.raises(FileNotFoundError, match="tokenizer_config.json"):
        Tokenizer.from_pretrained(tmp_path)


def test_unresolvable_token_ids_raises(tmp_path):
    """Token string not in vocabulary → ValueError."""
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token": "<NOT_IN_VOCAB>", "eos_token": "<ALSO_NOT>"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        Tokenizer.from_pretrained(tmp_path)


def test_bool_bos_token_id_raises(tmp_path):
    """bool is a subclass of int — must be rejected to avoid silent wrong behaviour."""
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token_id": True, "eos_token_id": EOS_ID}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bool"):
        Tokenizer.from_pretrained(tmp_path)


def test_negative_bos_token_id_raises(tmp_path):
    """Negative token ID is out of range and must be rejected."""
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token_id": -1, "eos_token_id": EOS_ID}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="out of range"):
        Tokenizer.from_pretrained(tmp_path)


def test_out_of_range_eos_token_id_raises(tmp_path):
    """Token ID >= vocab_size must be rejected."""
    _make_hf_tokenizer().save(str(tmp_path / "tokenizer.json"))
    config = {"bos_token_id": BOS_ID, "eos_token_id": 99999}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="out of range"):
        Tokenizer.from_pretrained(tmp_path)


def test_invalid_config_json_token_id_raises(tmp_path):
    """Fallback IDs from config.json are validated the same way as tokenizer_config IDs."""
    _make_hf_tokenizer_without_post_processor().save(str(tmp_path / "tokenizer.json"))
    tokenizer_config = {"bos_token": None, "eos_token": "[EOS]"}
    model_config = {"bos_token_id": True}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tmp_path / "config.json").write_text(json.dumps(model_config))

    with pytest.raises(ValueError, match="bool"):
        Tokenizer.from_pretrained(tmp_path)


# ---------------------------------------------------------------------------
# BOS / EOS token IDs
# ---------------------------------------------------------------------------

def test_bos_token_id_int_config(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    assert tok.bos_token_id == BOS_ID


def test_eos_token_id_int_config(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    assert tok.eos_token_id == EOS_ID


def test_bos_eos_str_config(tok_path_str_config):
    tok = Tokenizer.from_pretrained(tok_path_str_config)
    assert tok.bos_token_id == BOS_ID
    assert tok.eos_token_id == EOS_ID


def test_bos_eos_dict_config(tok_path_dict_config):
    tok = Tokenizer.from_pretrained(tok_path_dict_config)
    assert tok.bos_token_id == BOS_ID
    assert tok.eos_token_id == EOS_ID


def test_bos_from_config_json_and_eos_from_tokenizer_config(tok_path_qwen3_style_config):
    """Qwen3-style metadata resolves BOS and EOS from different files."""
    tok = Tokenizer.from_pretrained(tok_path_qwen3_style_config)
    assert tok.bos_token_id == BOS_ID
    assert tok.eos_token_id == EOS_ID


def test_bos_eos_are_ints(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    assert isinstance(tok.bos_token_id, int)
    assert isinstance(tok.eos_token_id, int)


# ---------------------------------------------------------------------------
# vocab_size
# ---------------------------------------------------------------------------

def test_vocab_size(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    assert tok.vocab_size == VOCAB_SIZE


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def test_encode_returns_list_of_ints(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = tok.encode("hello world", add_special_tokens=False)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_known_tokens(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = tok.encode("hello world", add_special_tokens=False)
    assert ids == [CONTENT_VOCAB["hello"], CONTENT_VOCAB["world"]]


def test_encode_add_special_tokens_true_prepends_bos(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = tok.encode("hello", add_special_tokens=True)
    assert ids[0] == tok.bos_token_id


def test_encode_add_special_tokens_true_longer_than_false(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids_with = tok.encode("hello", add_special_tokens=True)
    ids_without = tok.encode("hello", add_special_tokens=False)
    assert len(ids_with) == len(ids_without) + 1


def test_encode_add_special_tokens_false_no_bos(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = tok.encode("hello world", add_special_tokens=False)
    assert BOS_ID not in ids


def test_qwen3_style_encode_does_not_prepend_bos(tok_path_qwen3_style_config):
    """add_special_tokens=True follows tokenizer.json; Qwen3 plain prompts add no BOS."""
    tok = Tokenizer.from_pretrained(tok_path_qwen3_style_config)
    ids = tok.encode("hello", add_special_tokens=True)
    assert ids == [CONTENT_VOCAB["hello"]]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def test_decode_returns_str(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    text = tok.decode([CONTENT_VOCAB["hello"], CONTENT_VOCAB["world"]])
    assert isinstance(text, str)


def test_decode_known_ids(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    text = tok.decode([CONTENT_VOCAB["hello"], CONTENT_VOCAB["world"]], skip_special_tokens=True)
    assert "hello" in text and "world" in text


def test_decode_skip_special_tokens_removes_bos(tok_path_int_config):
    """skip_special_tokens=True filters tokens registered as special via add_special_tokens."""
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = [tok.bos_token_id, CONTENT_VOCAB["hello"]]
    assert "[BOS]" not in tok.decode(ids, skip_special_tokens=True)


def test_decode_skip_special_tokens_false_keeps_bos(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    ids = [tok.bos_token_id, CONTENT_VOCAB["hello"]]
    assert "[BOS]" in tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_encode_decode_round_trip(tok_path_int_config):
    tok = Tokenizer.from_pretrained(tok_path_int_config)
    original = "hello world"
    ids = tok.encode(original, add_special_tokens=False)
    recovered = tok.decode(ids, skip_special_tokens=True)
    assert "hello" in recovered and "world" in recovered


# ---------------------------------------------------------------------------
# Slow smoke tests (require local model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tokenizer_smoke():
    """Load real Llama-3.2-1B tokenizer and verify encode/decode round-trip."""
    import os
    model_path = os.environ.get("MODEL_PATH", "./models/llama-3.2-1b")
    tok = Tokenizer.from_pretrained(model_path)

    assert isinstance(tok.bos_token_id, int)
    assert isinstance(tok.eos_token_id, int)
    assert tok.vocab_size == 128256

    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=True)
    assert ids[0] == tok.bos_token_id

    ids_no_bos = tok.encode(prompt, add_special_tokens=False)
    assert len(ids) == len(ids_no_bos) + 1

    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "France" in decoded


@pytest.mark.slow
def test_tokenizer_eos_in_decode():
    """EOS token ID decodes to a non-empty string when skip_special_tokens=False."""
    import os
    model_path = os.environ.get("MODEL_PATH", "./models/llama-3.2-1b")
    tok = Tokenizer.from_pretrained(model_path)
    text = tok.decode([tok.eos_token_id], skip_special_tokens=False)
    assert isinstance(text, str)


@pytest.mark.slow
def test_qwen3_tokenizer_smoke():
    """Load real Qwen3-0.6B tokenizer metadata and verify plain prompt mode."""
    import os
    model_path = os.environ.get("QWEN_MODEL_PATH", "./models/qwen3-0.6b")
    tok = Tokenizer.from_pretrained(model_path)

    assert isinstance(tok.bos_token_id, int)
    assert isinstance(tok.eos_token_id, int)
    assert tok.vocab_size >= 151646

    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=True)
    ids_no_specials = tok.encode(prompt, add_special_tokens=False)

    # Qwen3-0.6B uses add_bos_token=false; tokenizers follows tokenizer.json
    # and does not synthesize a BOS token for plain prompt-to-completion mode.
    assert ids == ids_no_specials

    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "France" in decoded
