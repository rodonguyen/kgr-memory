"""Local chat page. The model is who you talk to; memory calls stream beside the reply."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from kgr_memory.chat_turn import CHAT_SEARCH_ROUNDS, load_locomo_sample, run_turn
from kgr_memory.graphiti_memory import add_text, build_graphiti, ensure_indices, search_facts
from kgr_memory.live import decide_live, plan_live, reply_live

HOST = "0.0.0.0"
PORT = 8765
_PAGE = Path(__file__).resolve().parent / "static" / "chat.html"


class Runtime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self._graphiti = None
        self._lock = threading.Lock()

    def stream(self, payload: dict, write) -> None:
        if not self._lock.acquire(blocking=False):
            write({"type": "error", "text": "A turn is already running."})
            return
        events: queue.Queue = queue.Queue()

        async def go() -> None:
            try:
                graphiti = await self._client()

                async def search(query: str):
                    return await search_facts(graphiti, query, payload["group_id"])

                async def store(memory: str) -> None:
                    await add_text(graphiti, memory, payload["group_id"])

                async for event in run_turn(
                    payload["text"],
                    plan=plan_live,
                    search=search,
                    store=store,
                    decide=decide_live,
                    answer=reply_live,
                    history=payload.get("history") or [],
                    prompt_name=payload.get("prompt") or "balanced",
                    max_rounds=CHAT_SEARCH_ROUNDS,
                ):
                    events.put(event)
            except Exception as exc:
                events.put({"type": "error", "text": str(exc)})
            finally:
                events.put(None)

        future = asyncio.run_coroutine_threadsafe(go(), self.loop)
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                write(event)
        finally:
            future.result()
            self._lock.release()

    async def _client(self):
        if self._graphiti is None:
            self._graphiti = build_graphiti()
            await ensure_indices(self._graphiti)
        return self._graphiti


def serve(port: int = PORT, host: str = HOST) -> None:
    runtime = Runtime()
    page = _PAGE.read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/chat"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/locomo":
                body = json.dumps(load_locomo_sample()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/api/turn":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = (payload.get("text") or "").strip()
            group_id = (payload.get("group_id") or "").strip()
            if not text or not group_id:
                self.send_error(400, "text and group_id are required")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def write(event: dict) -> None:
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            runtime.stream(payload, write)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[chat] {self.address_string()} {fmt % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"KGR chat at http://127.0.0.1:{port} and http://100.116.98.15:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
