import time

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


class BrowserController:
    def __init__(self):
        # Connect to the running Brave instance over CDP
        self.browser = None
        self.context = None

    def connect(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.connect_over_cdp(
            "http://localhost:9222"
        )
        self.context = self.browser.contexts[0]

    def open_classlist(self):
        if not self.browser or not self.context:
            raise Exception("Browser not connected. Call connect() first.")

        new_page = self.context.new_page()
        new_page.goto("https://e.huawei.com/en/talent/usercenter/#/home/myclass-list")
        print(f"Opened new tab: {new_page.title()}")
        new_page.bring_to_front()

        self.classes_page = new_page

    def goto_class(self, class_index: int) -> Page:
        if not self.browser or not self.context:
            raise Exception("Browser not connected. Call connect() first.")

        class_ = self.classes_page.locator(".lg.sm").nth(class_index)
        class_.wait_for(state="visible", timeout=10000)

        try:
            with self.classes_page.expect_popup(timeout=10000) as popup_info:
                class_.click()
            latest_tab = popup_info.value
        except PlaywrightTimeoutError:
            # Some flows reuse the same tab. Fall back to the latest available page.
            latest_tab = self.context.pages[-1]

        latest_tab.wait_for_load_state("domcontentloaded")
        latest_tab.bring_to_front()

        button = latest_tab.get_by_text("To Study", exact=True)
        button.scroll_into_view_if_needed()
        button.wait_for(state="visible", timeout=10000)

        try:
            with latest_tab.expect_popup(timeout=10000) as popup_info:
                button.click()
            study_tab = popup_info.value
        except PlaywrightTimeoutError:
            # If no popup is created, continue in current/latest tab.
            study_tab = self.context.pages[-1]

        study_tab.wait_for_load_state("domcontentloaded")
        study_tab.bring_to_front()

        return study_tab

    def cleanup(self):
        if self.browser:
            self.browser.close()
        try:
            self.playwright.stop()
        except Exception as e:
            print(f"Error stopping Playwright: {e}")


if __name__ == "__main__":
    try:
        controller = BrowserController()
        controller.connect()
        controller.open_classlist()
        time.sleep(4)
        print(controller.context.pages[-1].viewport_size)
        # course_page = controller.goto_class(0)
        # course_page.wait_for_load_state("domcontentloaded")

        input("Press Enter to close the browser...")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        controller.cleanup()
