from langchain_anthropic import ChatAnthropic

from core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def create_chat_model() -> ChatAnthropic:
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to the project .env file."
        )
    return ChatAnthropic(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL,
    )
