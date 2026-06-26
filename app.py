"""
Chainlit GUI for BiomedAgent.
=============================
Thin web front-end over the existing LangGraph agent in `unified_agent.py`.
Importing that module is side-effect-safe (the REPL is guarded by __main__),
so we reuse the exact same `agent`, tools, and routing prompt here.

Run:
    chainlit run app.py -w      # -w = auto-reload while editing

Each browser session maps to its own LangGraph `thread_id`, so the agent's
MemorySaver gives every visitor independent multi-turn memory. Tool calls are
surfaced live in the UI via Chainlit's LangChain callback handler — useful here
because correct tool routing is the core of this agent.
"""

import atexit

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk

from unified_agent import agent, LLM_MODEL, unload_llm


@cl.on_app_shutdown
async def _free_vram():
    """Drop the model from Ollama's VRAM when the server stops.

    Fires on ASGI lifespan shutdown, which uvicorn triggers on a graceful
    Ctrl+C / SIGTERM — so stopping `chainlit run` frees the VRAM. atexit alone
    is unreliable here because uvicorn doesn't always exit through it.
    """
    unload_llm()


# Belt-and-suspenders for any exit path that skips the lifespan shutdown.
atexit.register(unload_llm)


class ThinkFilter:
    """Strip <think>...</think> reasoning spans from a streamed token feed.

    Tokens arrive in arbitrary chunks, so an opening/closing tag can be split
    across two calls (e.g. "<thi" then "nk>"). We hold back any suffix that
    could be the start of a tag until we've seen enough to decide.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.in_think = False
        self.buf = ""

    @staticmethod
    def _safe_len(buf: str, tag: str) -> int:
        """Index up to which `buf` is safe to act on, holding back a partial tag."""
        for k in range(len(tag) - 1, 0, -1):
            if buf.endswith(tag[:k]):
                return len(buf) - k
        return len(buf)

    def feed(self, text: str) -> str:
        self.buf += text
        out = []
        while self.buf:
            if not self.in_think:
                idx = self.buf.find(self.OPEN)
                if idx == -1:
                    safe = self._safe_len(self.buf, self.OPEN)
                    out.append(self.buf[:safe])
                    self.buf = self.buf[safe:]
                    break
                out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(self.OPEN):]
                self.in_think = True
            else:
                idx = self.buf.find(self.CLOSE)
                if idx == -1:
                    self.buf = self.buf[self._safe_len(self.buf, self.CLOSE):]
                    break
                self.buf = self.buf[idx + len(self.CLOSE):]
                self.in_think = False
        return "".join(out)

    def flush(self) -> str:
        """Emit any trailing non-reasoning text held back at end of stream."""
        out = "" if self.in_think else self.buf
        self.buf = ""
        return out


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            f"**BiomedAgent** — running `{LLM_MODEL}` locally via Ollama.\n\n"
            "Ask a biomedical research question, e.g. "
            "_“What diseases are associated with TP53?”_ or "
            "_“Which cell types express CD8A in humans?”_"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    answer = cl.Message(content="")
    think = ThinkFilter()

    config = {
        "configurable": {"thread_id": cl.context.session.id},
        "recursion_limit": 50,
        # Renders each tool call (name, input, output) as a collapsible step.
        "callbacks": [cl.LangchainCallbackHandler()],
    }

    async for chunk, _meta in agent.astream(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        stream_mode="messages",
    ):
        # Stream only assistant text; tool/intermediate messages have no content
        # to stream (they carry tool_calls instead) and are shown as steps.
        # ThinkFilter drops <think>...</think> reasoning the model may emit inline.
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            visible = think.feed(chunk.content)
            if visible:
                await answer.stream_token(visible)

    trailing = think.flush()
    if trailing:
        await answer.stream_token(trailing)

    await answer.send()
