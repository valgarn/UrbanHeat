"""Test script running all key municipal planner questions through the ADK Agent."""

import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import agent.agent as ag

QUESTIONS = [
    "We can fund four cooling centers and one mobile hydration unit this summer in Dallas. Where should they go?",
    "Why is the intersection at latitude 32.7834, longitude -96.8623 so much hotter than average? What features drive that?",
    "How often does Dallas actually hit dangerous heat above 95°F and 100°F? Is it getting worse based on station history?",
    "Run an equity audit on our top recommended sites compared to the Dallas citywide baseline."
]

async def run_scenario(question: str):
    print("=" * 80)
    print(f"PLANNER QUESTION: {question}")
    print("=" * 80)
    
    session_service = InMemorySessionService()
    runner = Runner(app_name="dallas_heat_planner", agent=ag.root_agent, session_service=session_service)
    session = await session_service.create_session(app_name="dallas_heat_planner", user_id="planner_dana")
    
    msg = types.Content(role="user", parts=[types.Part.from_text(text=question)])
    async for event in runner.run_async(session_id=session.id, user_id="planner_dana", new_message=msg):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text, end="", flush=True)
                elif part.function_call:
                    print(f"\n[Invoking Tool: {part.function_call.name}({part.function_call.args})]\n")
    print("\n\n")

async def main():
    for q in QUESTIONS:
        await run_scenario(q)

if __name__ == "__main__":
    asyncio.run(main())
