"""client_ext attaches its helpers at import, and a patch that cannot be
attached raises instead of leaving the method silently missing."""

from __future__ import annotations

import importlib
import inspect

import pytest

import quantum_sdk
import quantum_sdk.client_ext as client_ext
from quantum_sdk import AsyncClient, Client

HELPERS = ["remesh", "rig", "animate", "retexture", "realtime_session_with"]


@pytest.mark.parametrize("name", HELPERS)
def test_helpers_are_attached_to_both_clients(name: str) -> None:
    sync, async_ = getattr(Client, name), getattr(AsyncClient, name)
    assert sync.__module__ == "quantum_sdk.client_ext"
    assert async_.__module__ == "quantum_sdk.client_ext"
    assert not inspect.iscoroutinefunction(sync)
    assert inspect.iscoroutinefunction(async_)


def test_patch_table_covers_every_helper() -> None:
    assert sorted(client_ext.PATCHES[Client]) == sorted(HELPERS)
    assert sorted(client_ext.PATCHES[AsyncClient]) == sorted(HELPERS)


def test_install_refuses_to_shadow_a_real_method() -> None:
    real_chat = Client.chat

    with pytest.raises(RuntimeError, match=r"Client\.chat.*quantum_sdk\.client"):
        client_ext._install({Client: {"chat": lambda self: None}})

    assert Client.chat is real_chat


def test_refused_install_attaches_nothing() -> None:
    """The check runs over every name first, so nothing is half-applied."""
    with pytest.raises(RuntimeError):
        client_ext._install({Client: {"zz_probe_helper": lambda self: None, "chat": lambda self: None}})

    assert not hasattr(Client, "zz_probe_helper")


def test_a_broken_patch_raises_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate client.py growing a real `remesh`: re-importing client_ext
    must fail, not quietly replace it or skip the rest of the patches."""

    def remesh(self):  # stands in for a method defined in client.py
        return "client.py's remesh"

    remesh.__module__ = "quantum_sdk.client"
    monkeypatch.setattr(Client, "remesh", remesh)

    with pytest.raises(RuntimeError, match=r"Client\.remesh"):
        importlib.reload(client_ext)

    assert Client.remesh is remesh


def test_package_import_does_not_swallow_client_ext_errors() -> None:
    """__init__ imports client_ext bare: no try/except stands between a
    failing patch and the caller's `import quantum_sdk`."""
    source = inspect.getsource(quantum_sdk)
    line = next(ln for ln in source.splitlines() if "import client_ext" in ln)
    assert line.startswith("from . import client_ext")
    assert "try:" not in inspect.getsource(client_ext)
