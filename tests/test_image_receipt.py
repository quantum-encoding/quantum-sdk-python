"""Image receipt fields — decode tests (no network).

Covers revised_prompt and usage on the image response, which the sync
/images/generate and /images/edit routes and the async image job results all
emit in the same shape. Wire contract source of truth: backend
routes_media.go (imageGenResponse, imageUsage, imageUsageOf).
"""

from __future__ import annotations

from quantum_sdk import ImageResponse


def test_image_receipt_carries_the_rewrite_and_the_usage() -> None:
    """A token-priced generation: the usage is the whole audit, and the
    revised prompt is the text the picture was actually made from."""
    resp = ImageResponse.from_dict(
        {
            "images": [{"base64": "iVBOR", "format": "png", "index": 0}],
            "model": "gpt-image-2",
            "cost_ticks": 527000000,
            "balance_after": 91,
            "request_id": "qai_req_f372518d-447",
            "revised_prompt": (
                "A golden rubber duck wearing a black silk top hat, studio lit"
            ),
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 1580,
                "total_tokens": 1700,
            },
        }
    )

    assert resp.revised_prompt == (
        "A golden rubber duck wearing a black silk top hat, studio lit"
    )
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 120
    assert resp.usage.completion_tokens == 1580
    assert resp.usage.total_tokens == 1700
    assert resp.balance_after == 91


def test_image_receipt_without_them_reports_none_not_empty() -> None:
    """A flat-priced model rewrites nothing and reports no tokens, so both
    stay None rather than becoming an empty string and zeros that would look
    measured. A gateway predating these fields must still decode."""
    resp = ImageResponse.from_dict(
        {
            "images": [],
            "model": "grok-imagine-image-2.0",
            "cost_ticks": 500000000,
            "balance_after": 88,
            "request_id": "qai_req_9c03b229-d09",
        }
    )

    assert resp.revised_prompt is None, "a rewrite was invented"
    assert resp.usage is None, "a usage block was invented"


def test_async_image_job_result_decodes_the_same_receipt() -> None:
    """The async job result carries the same envelope the sync route emits,
    so a job caller can check the same bill. It carries no balance_after."""
    resp = ImageResponse.from_dict(
        {
            "images": [{"base64": "iVBOR", "format": "png", "index": 0}],
            "model": "gpt-image-2",
            "cost_ticks": 162800000,
            "request_id": "qai_req_4a1e77c0-2b8",
            "revised_prompt": "A golden rubber duck, softbox lit",
            "usage": {"prompt_tokens": 96, "total_tokens": 610},
        }
    )

    assert resp.revised_prompt == "A golden rubber duck, softbox lit"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 96
    assert resp.usage.total_tokens == 610
    assert resp.balance_after is None
