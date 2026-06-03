import os
import time
import logging

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.tools import ToolBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# System prompt: tells Claude what it is and how to route between tools.
SYSTEM_PROMPT = """\
You are a helpful assistant for the US Census "Adult Income" domain. You can \
both answer questions about income, demographics, education and socioeconomic \
factors, and make income predictions using a machine-learning model trained in \
Homework 1.

You have access to three tools:

1. retrieve_knowledge — searches a domain knowledge base of documents (US \
personal income, educational attainment, the gender pay gap, income \
inequality, algorithmic bias, and the dataset description). Use this for \
factual or conceptual "what / why / how" questions about the domain.

2. predict_income — runs the Homework 1 model to predict whether a person's \
annual income exceeds $50K, given their census features. Use this when the \
user supplies (or has supplied earlier in the conversation) specific \
demographic / employment values and wants a prediction. The tool requires all \
14 census fields; if the user has not provided some of them, ask for the \
missing values rather than inventing them.

3. dataset_stats — returns summary statistics for a column of the Homework 1 \
dataset (e.g. average age, class balance, distribution of occupations). Use \
this when the user asks about the dataset itself.

Guidelines:
- Decide which tool to use (or none) based on the user's message. Never ask \
the user to name a tool.
- For follow-up questions, reuse the values and context from earlier turns. If \
the user changes one field ("what if the discount were higher"), re-run the \
prediction with the updated value while keeping the rest.
- When you answer from the knowledge base, ground your answer in the retrieved \
passages. If the knowledge base has nothing relevant, say so rather than \
guessing.
- Keep answers concise and conversational.\
"""

# A fast, inexpensive default that handles tool routing well. Override with the
# GOOGLE_MODEL environment variable.
DEFAULT_MODEL = "gemini-2.5-flash"


class ConversationalAgent:
    """
    Builds and serves the LangGraph agent.

    Lifecycle
    ─────────
        agent = ConversationalAgent().build()      # load tools + compile graph
        reply = agent.chat("...", session_id="user_001")

    `build()` is expensive (loads the vector store, HW1 artifacts and dataset),
    so build once and reuse the instance across requests.
    """

    def __init__(self, model_name: str = None, temperature: float = 0.0):
        self.model_name = model_name or os.getenv("GOOGLE_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.tools = None
        self.llm = None
        self.checkpointer = None
        self.agent = None  # compiled LangGraph graph

    def build(self):
        """Load tools, instantiate the LLM, and compile the agent graph."""
        load_dotenv()  # pick up GOOGLE_API_KEY / GOOGLE_MODEL from .env

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and "
                "add your Google API key (or export it in your shell)."
            )

        # Log a MASKED fingerprint of the key that was actually loaded. Note
        # load_dotenv() does not override an env var already set in the shell /
        # IDE run config, so this is the only reliable way to confirm *which*
        # key the agent is using after you swap keys in .env (you must also
        # restart the process — the key is read once, here at startup).
        key_fp = f"len={len(api_key)} tail=...{api_key[-4:]}"

        logging.info("=" * 60)
        logging.info("Building conversational agent...")
        logging.info(f"Model: {self.model_name} | temperature: {self.temperature}")
        logging.info(f"GOOGLE_API_KEY loaded: {key_fp}")
        logging.info("=" * 60)

        # Tools (loads the vector store, HW1 model, dataset).
        self.tools = ToolBuilder().run_pipeline()

        # LLM.
        # thinking_budget=0 disables Gemini 2.5's "thinking" tokens. This task
        # is tool routing + concise answers, which doesn't need extended
        # reasoning; leaving thinking on makes some models (notably
        # gemini-2.5-flash-lite) spend their whole budget thinking and return
        # an empty answer. Disabling it is faster, cheaper, and reliable.
        # max_retries=1: by default the google-genai client retries a 429 up to
        # ~6 times with exponential backoff (~30s) before surfacing the error,
        # which makes a quota-exhausted request hang. One try fails fast so the
        # clean 429 reaches the client in a couple of seconds.
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            thinking_budget=0,
            max_retries=1,
        )

        # Per-session memory (within-process; resets on restart, which is
        #    all the assignment requires).
        self.checkpointer = MemorySaver()

        # Compile the ReAct agent graph.
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )

        logging.info("Conversational agent ready.")
        return self

    # Helpers
    @staticmethod
    def _extract_text(message):
        """
        Pull plain text out of an assistant message (or streamed chunk). LLM
        responses can be a plain string or a list of content blocks; handle both.
        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts).strip()
        return str(content)

    # Public API
    def chat(self, message: str, session_id: str):
        """
        Send one user message and return the agent's text reply. Conversation
        history is keyed by session_id, so follow-up turns see prior context.
        """
        if self.agent is None:
            raise RuntimeError("Agent not built. Call build() first.")

        config = {"configurable": {"thread_id": session_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return self._extract_text(result["messages"][-1])

    async def astream_chat(self, message: str, session_id: str):
        """
        Async generator that yields the agent's reply token-by-token, for the
        streaming API endpoint (Task 6). Conversation history is keyed by
        session_id, exactly like `chat`.

        We stream in "messages" mode and forward only the textual chunks of the
        assistant's answer. Intermediate tool-routing chunks carry tool calls
        rather than text, so filtering on non-empty text naturally skips them.
        """
        if self.agent is None:
            raise RuntimeError("Agent not built. Call build() first.")

        config = {"configurable": {"thread_id": session_id}}
        yielded_any = False
        async for chunk, _metadata in self.agent.astream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode="messages",
        ):
            # Only forward chunks emitted by the chat model (assistant text),
            # not ToolMessage outputs that also flow through the graph.
            if isinstance(chunk, AIMessage):
                text = self._extract_text(chunk)
                if text:
                    yielded_any = True
                    yield text

        # Robustness fallback: on some turns (typically when a tool was called)
        # Gemini streams only the tool-call chunks and delivers the final answer
        # without incremental text, so the loop above yields nothing and the
        # client would otherwise see an empty reply. In that case, read the
        # finished answer from the checkpointed state and emit it in one go.
        # This costs no extra LLM call — the turn has already completed — and
        # guarantees /chat/stream returns the same text as /chat.
        if not yielded_any:
            state = await self.agent.aget_state(config)
            messages = state.values.get("messages", [])
            if messages:
                text = self._extract_text(messages[-1])
                if text:
                    yield text


# CLI smoke test — `python -m src.agent`
# Runs a short scripted conversation to verify tool routing + memory.

def main():
    agent = ConversationalAgent().build()
    session = "smoke_test"

    turns = [
        # RAG retrieval
        "How does education relate to income?",
        # Prediction (full feature set)
        ('Would this person earn over $50K? age 39, State-gov workclass, '
         'fnlwgt 77516, Bachelors education, education_num 13, Never-married, '
         'Adm-clerical occupation, Not-in-family relationship, White, Male, '
         'capital_gain 2174, capital_loss 0, 40 hours per week, United-States.'),
        # Follow-up that relies on memory (change one field)
        "What if they worked 60 hours per week instead?",
        # Dataset stats (bonus tool)
        "What is the average age in the dataset?",
    ]

    # Each ReAct turn makes ~2 LLM calls. Gemini's free tier allows only 5
    # requests/minute, so pause between turns to keep the scripted test under
    # the limit. Real API usage (one message at a time) doesn't need this.
    pause_s = float(os.getenv("SMOKE_TEST_PAUSE_S", "20"))

    for i, msg in enumerate(turns):
        print("\n" + "=" * 72)
        print(f"USER: {msg}")
        print("-" * 72)
        print(f"AGENT: {agent.chat(msg, session_id=session)}")
        if i < len(turns) - 1 and pause_s > 0:
            time.sleep(pause_s)
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
