# main.py

from windows_use.llms.google import ChatGoogle
from windows_use.agent import Agent, Browser
from dotenv import load_dotenv
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"
load_dotenv()


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogle(model="gemini-3-flash-preview", api_key=api_key, temperature=0.7)
    agent = Agent(
        llm=llm, browser=Browser.EDGE, use_vision=True, auto_minimize=False, max_steps=1
    )

    print("\n" + "=" * 60)
    print("🎓 TEACHING MODE - Interactive Tutorial")
    print("=" * 60)

    # Get the user's learning goal
    user_query = input("\n📝 What would you like to learn? \n> ")

    # Build the teaching prompt
    teaching_context = f"""
IMPORTANT: You are in TEACHING MODE - an interactive, step-by-step tutorial.

RULES:
- DO NOT click, type, or interact with anything - only observe and guide
- Give exactly ONE clear instruction at a time
- Point out specific UI elements, their locations, and what they look like
- After giving your instruction, use the Done Tool to pause and let the user complete it
- Keep instructions concise but helpful

USER'S LEARNING GOAL: {user_query}

Look at the screen and give the FIRST instruction to get started.
"""

    step_number = 1

    while True:
        print(f"\n{'─'*60}")
        print(f"📚 Step {step_number}")
        print("─" * 60)

        # Agent observes and gives one instruction
        result = agent.invoke(teaching_context)

        if result.error:
            print(f"\n⚠️ Agent encountered an issue: {result.error}")

        # Wait for user input before continuing
        print("\n" + "─" * 60)
        user_input = (
            input("✨ Press Enter when done, or type 'quit' to exit: ").strip().lower()
        )

        if user_input in ["quit", "exit", "q", "done"]:
            print("\n🎉 Great session! Keep practicing!")
            break

        # Prepare prompt for next step - agent checks progress and gives next instruction
        teaching_context = f"""
TEACHING MODE - Step {step_number + 1}

The user says they've completed the previous step. 
- Look at the screen to verify their progress
- If they did it correctly, acknowledge briefly and give the NEXT instruction
- If something looks wrong, gently point out what to fix
- When the entire goal is achieved, congratulate them and use Done Tool with "Tutorial complete!"

USER'S ORIGINAL GOAL: {user_query}

Check their progress and give the next instruction (or confirm completion).
"""
        step_number += 1


if __name__ == "__main__":
    main()
