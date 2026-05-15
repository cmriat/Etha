"""Stress chat against the vLLM OpenAI server, gated by the control store.

Reads `vllm_ready` / `vllm_version` to skip requests during sync and tag
each reply with the live weight version. Bumps `chat_count` after every
successful completion — trainer rank 0 watches that counter to gate the
next sync round, so the demo runs at the chat's pace.
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import CONFIG, open_control_store
from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPTS = (
    "The capital of France is",
    "Q: What is 2 + 2?\nA:",
    "Once upon a time,",
)


def _wait_for_server(client: OpenAI, timeout: float = 1200.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.models.list()
            return
        except Exception as e:
            logger.info("waiting for vllm server: %s", e)
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready after {timeout}s")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [chat] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    store = open_control_store()
    logger.info("control store connected")

    client = OpenAI(base_url=f"http://{CONFIG.vllm_http_host}:{CONFIG.vllm_http_port}/v1", api_key="dummy")
    _wait_for_server(client)
    logger.info("vllm server is up at %s:%d", CONFIG.vllm_http_host, CONFIG.vllm_http_port)

    round_idx = 0
    chat_count = 0
    store.set("chat_count", "0")
    while CONFIG.chat_rounds == 0 or round_idx < CONFIG.chat_rounds:
        ready = store.get("vllm_ready")
        if ready != b"1":
            time.sleep(CONFIG.not_ready_sleep)
            continue

        version_raw = store.get("vllm_version")
        version = int(version_raw) if version_raw else 0

        round_idx += 1
        for prompt in PROMPTS:
            try:
                resp = client.completions.create(
                    model=CONFIG.model_id,
                    prompt=prompt,
                    max_tokens=32,
                    temperature=0.0,
                )
                logger.info(
                    "v=%d round=%d prompt=%r reply=%r",
                    version,
                    round_idx,
                    prompt,
                    resp.choices[0].text,
                )
                chat_count += 1
                store.set("chat_count", str(chat_count))
                logger.debug("chat_count -> %d", chat_count)
            except Exception:
                logger.exception("chat round=%d prompt=%r failed", round_idx, prompt)
        time.sleep(CONFIG.chat_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
