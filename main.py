# main.py

from windows_use.llms.google import ChatGoogle
from windows_use.agent import Agent, Browser
from dotenv import load_dotenv
from pathlib import Path
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
load_dotenv()

GOAL_FILE = Path(__file__).parent / "user-goal.md"


def read_user_goal():
    """Read the current goal from user-goal.md file."""
    if GOAL_FILE.exists():
        return GOAL_FILE.read_text(encoding="utf-8").strip()
    return "No goal specified. Please add your learning goal to user-goal.md"


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogle(model="gemini-3-flash-preview", api_key=api_key, temperature=0.7)
    agent = Agent(
        llm=llm, browser=Browser.EDGE, use_vision=True, auto_minimize=False, max_steps=1
    )

    print("\n" + "=" * 60)
    print("🎓 TEACHING MODE - Interactive Tutorial")
    print("=" * 60)

    # Read initial goal from file
    user_goal = read_user_goal()
    print(
        f"\n📝 Goal loaded from user-goal.md:\n   {user_goal[:100]}{'...' if len(user_goal) > 100 else ''}"
    )

    step_number = 1

    while True:
        # Re-read goal each step (in case user updates it)
        user_goal = read_user_goal()

        print(f"\n{'─' * 60}")
        print(f"📚 Step {step_number}")
        print("─" * 60)

        # Build context based on step number
        if step_number == 1:
            teaching_context = f"""
IMPORTANT: You are in TEACHING MODE - an interactive, step-by-step tutorial.

RULES:
- You CAN use the Move Tool to point your cursor at UI elements to show the user where they are
- Do NOT click, type, drag, or make any changes - only observe, point, and guide
- Give exactly ONE clear instruction at a time
- Use Move Tool to hover over the element you're describing so the user can see where it is
- After pointing and explaining, use the Done Tool to pause and let the user do it themselves
- Keep instructions concise but helpful

USER'S LEARNING GOAL: {user_goal}

Look at the screen and give the FIRST instruction. Move your cursor to show them where to look.
"""
        else:
            teaching_context = f"""
TEACHING MODE - Step {step_number}

The user says they've completed the previous step.

RULES:
- You CAN use the Move Tool to point your cursor at UI elements to show the user where they are
- Do NOT click, type, drag, or make any changes - only observe, point, and guide
- Look at the screen to verify their progress
- If they did it correctly, acknowledge briefly and give the NEXT instruction
- If something looks wrong, gently point out what to fix
- Use Move Tool to hover over the next element they need to interact with
- When the entire goal is achieved, congratulate them and use Done Tool with "Tutorial complete!"

USER'S CURRENT GOAL: {user_goal}

Check their progress and give the next instruction (or confirm completion).
"""

        # Agent observes and gives one instruction
        result = agent.invoke(teaching_context)

        if result.error:
            print(f"\n⚠️ Agent encountered an issue: {result.error}")

        # Check if tutorial is complete
        if result.content and "tutorial complete" in result.content.lower():
            print("\n🎉 Tutorial complete! Great job!")
            break

        # Wait for user input before continuing
        print("\n" + "─" * 60)
        user_input = (
            input("✨ Press Enter when done, or type 'quit' to exit: ").strip().lower()
        )

        if user_input in ["quit", "exit", "q"]:
            print("\n🎉 Great session! Keep practicing!")
            break

        step_number += 1


if __name__ == "__main__":
    main()
