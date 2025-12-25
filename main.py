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
    agent = Agent(llm=llm, browser=Browser.EDGE, use_vision=True, auto_minimize=False)

    # Prepend teaching instructions to the user's query
    teaching_prompt = """
    IMPORTANT: You are in TEACHING MODE. 
    - DO NOT click or type anything but do look at their screen to monitor and guide their progress 
    - With a short plan in mind, explain the next step what a user should do
    - Point out UI elements and their locations
    - Watch the user's actions and continue giving them instructions until their goal is achieved
    
    User's question: You are a helpful and intelligent assistant and tutor, helping me with use and 
    learn software on my computer. Your current goal is to help me create a hard hitting 808 kick 
    synth using Serum 2. I'm trying to create a sound the style of artists like uk drill or brooklyn 
    drill who have that gritty, sweeping, warped, gliding sound.
    """
    agent.print_response(query=teaching_prompt)


if __name__ == "__main__":
    main()
