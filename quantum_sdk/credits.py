"""Credits — purchase credit packs, check balance, view tiers, dev program."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Types ──────────────────────────────────────────────────────────────

@dataclass
class CreditPack:
    """A credit pack available for purchase."""

    id: str = ""
    name: str | None = None
    price_usd: float = 0.0
    credit_ticks: int = 0
    description: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditPack:
        return cls(
            id=data.get("id", ""),
            name=data.get("name"),
            price_usd=data.get("price_usd", 0.0),
            credit_ticks=data.get("credit_ticks", 0),
            description=data.get("description"),
        )


@dataclass
class CreditPacksResponse:
    """Response from listing credit packs."""

    packs: list[CreditPack] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditPacksResponse:
        return cls(packs=[CreditPack.from_dict(p) for p in data.get("packs", [])])


@dataclass
class CreditPurchaseResponse:
    """Response from purchasing a credit pack."""

    checkout_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditPurchaseResponse:
        return cls(checkout_url=data.get("checkout_url", ""))


@dataclass
class CreditBalanceResponse:
    """Response from checking credit balance."""

    balance_ticks: int = 0
    balance_usd: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditBalanceResponse:
        return cls(
            balance_ticks=data.get("balance_ticks", 0),
            balance_usd=data.get("balance_usd", 0.0),
        )


@dataclass
class CreditTier:
    """A pricing tier."""

    name: str | None = None
    min_balance: int = 0
    discount_percent: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditTier:
        return cls(
            name=data.get("name"),
            min_balance=data.get("min_balance", 0),
            discount_percent=data.get("discount_percent", 0.0),
        )


@dataclass
class CreditTiersResponse:
    """Response from listing credit tiers."""

    tiers: list[CreditTier] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditTiersResponse:
        return cls(tiers=[CreditTier.from_dict(t) for t in data.get("tiers", [])])


@dataclass
class DevProgramApplyResponse:
    """Response from dev program application."""

    status: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevProgramApplyResponse:
        return cls(status=data.get("status", ""))


# ── Functions (sync) ──────────────────────────────────────────────────

def credit_packs_sync(client: Any) -> CreditPacksResponse:
    """List available credit packs (sync). No auth required."""
    data, _ = client._do_json("GET", "/qai/v1/credits/packs")
    return CreditPacksResponse.from_dict(data)


def credit_purchase_sync(client: Any, pack_id: str, success_url: str | None = None, cancel_url: str | None = None) -> CreditPurchaseResponse:
    """Purchase a credit pack (sync)."""
    body: dict[str, Any] = {"pack_id": pack_id}
    if success_url is not None:
        body["success_url"] = success_url
    if cancel_url is not None:
        body["cancel_url"] = cancel_url
    data, _ = client._do_json("POST", "/qai/v1/credits/purchase", body)
    return CreditPurchaseResponse.from_dict(data)


def credit_balance_sync(client: Any) -> CreditBalanceResponse:
    """Get credit balance (sync)."""
    data, _ = client._do_json("GET", "/qai/v1/credits/balance")
    return CreditBalanceResponse.from_dict(data)


def credit_tiers_sync(client: Any) -> CreditTiersResponse:
    """List credit tiers (sync). No auth required."""
    data, _ = client._do_json("GET", "/qai/v1/credits/tiers")
    return CreditTiersResponse.from_dict(data)


def dev_program_apply_sync(client: Any, use_case: str, company: str | None = None, expected_usd: float | None = None, website: str | None = None) -> DevProgramApplyResponse:
    """Apply for the developer program (sync)."""
    body: dict[str, Any] = {"use_case": use_case}
    if company is not None:
        body["company"] = company
    if expected_usd is not None:
        body["expected_usd"] = expected_usd
    if website is not None:
        body["website"] = website
    data, _ = client._do_json("POST", "/qai/v1/credits/dev-program", body)
    return DevProgramApplyResponse.from_dict(data)


# ── Functions (async) ─────────────────────────────────────────────────

async def credit_packs_async(client: Any) -> CreditPacksResponse:
    """List available credit packs (async). No auth required."""
    data, _ = await client._do_json("GET", "/qai/v1/credits/packs")
    return CreditPacksResponse.from_dict(data)


async def credit_purchase_async(client: Any, pack_id: str, success_url: str | None = None, cancel_url: str | None = None) -> CreditPurchaseResponse:
    """Purchase a credit pack (async)."""
    body: dict[str, Any] = {"pack_id": pack_id}
    if success_url is not None:
        body["success_url"] = success_url
    if cancel_url is not None:
        body["cancel_url"] = cancel_url
    data, _ = await client._do_json("POST", "/qai/v1/credits/purchase", body)
    return CreditPurchaseResponse.from_dict(data)


async def credit_balance_async(client: Any) -> CreditBalanceResponse:
    """Get credit balance (async)."""
    data, _ = await client._do_json("GET", "/qai/v1/credits/balance")
    return CreditBalanceResponse.from_dict(data)


async def credit_tiers_async(client: Any) -> CreditTiersResponse:
    """List credit tiers (async). No auth required."""
    data, _ = await client._do_json("GET", "/qai/v1/credits/tiers")
    return CreditTiersResponse.from_dict(data)


async def dev_program_apply_async(client: Any, use_case: str, company: str | None = None, expected_usd: float | None = None, website: str | None = None) -> DevProgramApplyResponse:
    """Apply for the developer program (async)."""
    body: dict[str, Any] = {"use_case": use_case}
    if company is not None:
        body["company"] = company
    if expected_usd is not None:
        body["expected_usd"] = expected_usd
    if website is not None:
        body["website"] = website
    data, _ = await client._do_json("POST", "/qai/v1/credits/dev-program", body)
    return DevProgramApplyResponse.from_dict(data)
