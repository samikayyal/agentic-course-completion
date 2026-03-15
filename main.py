from agent import Agent
from browser import BrowserController

if __name__ == "__main__":
    try:
        controller = BrowserController()
        controller.connect()
        controller.open_classlist()
        input(
            "The browser might prompt you to login or select a class. After you have done that, press Enter to continue..."
        )
        course_page = controller.goto_class(1)
        course_page.wait_for_load_state("domcontentloaded")

        agent = Agent(controller)
        agent.run_loop()
    except Exception as e:
        print(f"Error in main: {e}")
