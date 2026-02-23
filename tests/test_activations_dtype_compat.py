from __future__ import annotations

from typing import Any

import pytest

from ta_probe import activations


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = None
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = None


class _FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.to_device: str | None = None

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def to(self, device: str) -> _FakeModel:
        self.to_device = device
        return self


def test_load_model_prefers_dtype_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def fake_tokenizer_loader(*_args, **_kwargs) -> _FakeTokenizer:
        return tokenizer

    def fake_model_loader(*_args, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(
        activations.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        activations.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_loader,
    )

    loaded_model, loaded_tokenizer = activations.load_model_and_tokenizer(
        model_name_or_path="fake-model",
        device="cpu",
        dtype_name="float16",
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert len(calls) == 1
    assert "dtype" in calls[0]
    assert "torch_dtype" not in calls[0]
    assert model.eval_called is True
    assert model.to_device == "cpu"
    assert tokenizer.pad_token == tokenizer.eos_token


def test_load_model_falls_back_to_torch_dtype_on_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    def fake_tokenizer_loader(*_args, **_kwargs) -> _FakeTokenizer:
        return tokenizer

    def fake_model_loader(*_args, **kwargs):
        calls.append(kwargs)
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        return model

    monkeypatch.setattr(
        activations.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        activations.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_loader,
    )

    with pytest.warns(RuntimeWarning, match="Fell back to `torch_dtype`"):
        loaded_model, loaded_tokenizer = activations.load_model_and_tokenizer(
            model_name_or_path="fake-model",
            device="cpu",
            dtype_name="float16",
        )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert len(calls) == 2
    assert "dtype" in calls[0]
    assert "torch_dtype" in calls[1]
    assert model.eval_called is True
    assert model.to_device == "cpu"


def test_load_model_raises_clear_error_when_both_dtype_paths_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()

    def fake_tokenizer_loader(*_args, **_kwargs) -> _FakeTokenizer:
        return tokenizer

    def fake_model_loader(*_args, **kwargs):
        if "dtype" in kwargs:
            raise TypeError("dtype path failed")
        raise ValueError("torch_dtype path failed")

    monkeypatch.setattr(
        activations.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_loader,
    )
    monkeypatch.setattr(
        activations.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_loader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed loading model with both dtype compatibility paths",
    ) as exc:
        activations.load_model_and_tokenizer(
            model_name_or_path="fake-model",
            device="cpu",
            dtype_name="float16",
        )

    rendered = str(exc.value)
    assert "dtype path failed" in rendered
    assert "torch_dtype path failed" in rendered
