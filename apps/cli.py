"""Консольный чат для ручной проверки: python -m apps.cli clinic"""

import argparse
import logging
import sys

from core.agent import Agent
from core.config import get_settings
from core.llm import OpenAICompatLLM
from domains import build_domain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=["clinic"], nargs="?", default="clinic")
    parser.add_argument("--debug", action="store_true", help="показывать состояние после каждой реплики")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING)

    settings = get_settings()
    if not settings.llm_enabled:
        print("⚠️  LLM_API_KEY не задан в .env — работает только разбор по правилам", file=sys.stderr)
    llm = OpenAICompatLLM(settings) if settings.llm_enabled else None
    agent = Agent(build_domain(args.domain), llm, nlg=settings.llm_nlg)
    session = agent.new_session()
    print(f"Модель: {settings.llm_model if llm else 'нет (правила)'}. Команды: /state, /reset, /quit")

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/reset":
            session = agent.new_session()
            print("Диалог сброшен.")
            continue
        if text == "/state":
            print(session.state.model_dump_json(indent=2, exclude_none=True))
            continue
        print(agent.handle(session, text).reply)
        if args.debug:
            print(f"[{session.last_nlu}]", session.state.model_dump_json(exclude_none=True, exclude={"offered"}))


if __name__ == "__main__":
    main()
