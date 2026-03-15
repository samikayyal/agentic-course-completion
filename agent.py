import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import Content, Part

load_dotenv()
SCREEN_WIDTH = int(os.getenv("SCREEN_WIDTH", 1920))
SCREEN_HEIGHT = int(os.getenv("SCREEN_HEIGHT", 1080))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 200000))

CUSTOM_TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "click_at",
        "description": (
            "Clicks at a specific coordinate on the webpage. "
            "Coordinates are normalized to a 0-999 grid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Normalized x coordinate (0-999).",
                },
                "y": {
                    "type": "integer",
                    "description": "Normalized y coordinate (0-999).",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "multiple_clicks",
        "description": (
            "Clicks multiple coordinates in sequence. Receives a list of coordinate "
            "objects, each containing normalized x and y values on a 0-999 grid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "coordinates": {
                    "type": "array",
                    "description": "Ordered list of coordinates to click.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "Normalized x coordinate (0-999).",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Normalized y coordinate (0-999).",
                            },
                        },
                        "required": ["x", "y"],
                    },
                }
            },
            "required": ["coordinates"],
        },
    },
    {
        "name": "click_material_until_completed",
        "description": (
            "Clicks the single-step next arrow in reading materials repeatedly "
            "until the requested number of clicks is reached. Coordinates are "
            "normalized to a 0-999 grid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Normalized x coordinate (0-999) of the single next arrow.",
                },
                "y": {
                    "type": "integer",
                    "description": "Normalized y coordinate (0-999) of the single next arrow.",
                },
                "clicks": {
                    "type": "integer",
                    "description": "How many times to click the single next arrow.",
                },
            },
            "required": ["x", "y", "clicks"],
        },
    },
    {
        "name": "hover_at",
        "description": (
            "Moves the mouse to a specific coordinate to reveal hover interactions. "
            "Coordinates are normalized to a 0-999 grid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Normalized x coordinate (0-999).",
                },
                "y": {
                    "type": "integer",
                    "description": "Normalized y coordinate (0-999).",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "scroll_document",
        "description": "Scrolls the page viewport in a direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "description": "Scroll direction.",
                    "enum": ["up", "down", "left", "right"],
                }
            },
            "required": ["direction"],
        },
    },
    {
        "name": "scroll_at",
        "description": (
            "Scrolls at a specific coordinate, useful for nested scroll containers. "
            "Coordinates and magnitude use a 0-999 style normalized scale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "integer",
                    "description": "Normalized x coordinate (0-999).",
                },
                "y": {
                    "type": "integer",
                    "description": "Normalized y coordinate (0-999).",
                },
                "direction": {
                    "type": "string",
                    "description": "Scroll direction.",
                    "enum": ["up", "down", "left", "right"],
                },
                "magnitude": {
                    "type": "integer",
                    "description": "Scroll magnitude. Defaults to 800 if omitted.",
                },
            },
            "required": ["x", "y", "direction"],
        },
    },
    {
        "name": "wait_x_seconds",
        "description": "Pauses execution for the specified number of seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Number of seconds to wait (>= 0).",
                }
            },
            "required": ["seconds"],
        },
    },
]


def denormalize_x(x: int, width: int) -> int:
    """Convert normalized x coordinate (0-1000) to CSS pixel coordinate."""
    return int(x / 1000 * width)


def denormalize_y(y: int, height: int) -> int:
    """Convert normalized y coordinate (0-1000) to CSS pixel coordinate."""
    return int(y / 1000 * height)


class Agent:
    def __init__(self, browser_controller):
        self.browser_controller = browser_controller
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = os.getenv("GEMINI_MODEL", "")
        if not self.model_name:
            raise ValueError("GEMINI_MODEL environment variable is not set.")

        self.generate_content_config = genai.types.GenerateContentConfig(
            # We manually execute tool calls in this agent loop, so disable AFC.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tools=[
                types.Tool(function_declarations=CUSTOM_TOOL_DECLARATIONS),  # type: ignore
            ],
        )

        self.contents: list[Content] = []

    def _get_context_token_count(self) -> int:
        """Return current token count for `self.contents` using Gemini's tokenizer."""
        if not self.contents:
            return 0

        count_response = self.client.models.count_tokens(
            model=self.model_name,
            contents=self.contents,  # ty:ignore[invalid-argument-type]
        )

        token_count = getattr(count_response, "total_tokens", None)
        if token_count is None:
            token_count = getattr(count_response, "totalTokens", None)
        if token_count is None and isinstance(count_response, dict):
            token_count = count_response.get("total_tokens")
        if token_count is None and isinstance(count_response, dict):
            token_count = count_response.get("totalTokens")
        return int(token_count or 0)

    def _show_click_marker(self, page, x: int, y: int):
        """Render a short-lived visual marker at the click position."""
        page.evaluate(
            """
            ({ x, y }) => {
                const marker = document.createElement('div');
                marker.style.position = 'fixed';
                marker.style.left = `${x - 15}px`;
                marker.style.top = `${y - 15}px`;
                marker.style.width = '30px';
                marker.style.height = '30px';
                marker.style.border = '3px solid #ff2d55';
                marker.style.borderRadius = '9999px';
                marker.style.background = 'rgba(255, 45, 85, 0.15)';
                marker.style.boxShadow = '0 0 0 4px rgba(255, 45, 85, 0.25)';
                marker.style.zIndex = '2147483647';
                marker.style.pointerEvents = 'none';
                marker.style.transition = 'transform 0.35s ease, opacity 0.35s ease';
                marker.style.transform = 'scale(0.75)';
                marker.style.opacity = '1';
                document.body.appendChild(marker);

                requestAnimationFrame(() => {
                    marker.style.transform = 'scale(1.2)';
                    marker.style.opacity = '0.95';
                });

                setTimeout(() => {
                    marker.style.opacity = '0';
                    marker.style.transform = 'scale(1.6)';
                }, 220);

                setTimeout(() => marker.remove(), 650);
            }
            """,
            {"x": x, "y": y},
        )

    def _get_viewport_size(self, page) -> tuple[int, int]:
        """Read live viewport size from the page (CSS pixels)."""
        metrics = page.evaluate(
            """
            () => ({
                innerWidth: window.innerWidth,
                innerHeight: window.innerHeight
            })
            """
        )

        width = int(metrics.get("innerWidth") or 0)
        height = int(metrics.get("innerHeight") or 0)

        # Fallbacks preserve previous behavior if page metrics are unavailable.
        if width <= 0:
            print("Warning: Unable to read viewport width, using fallback.")
            width = SCREEN_WIDTH
        if height <= 0:
            print("Warning: Unable to read viewport height, using fallback.")
            height = SCREEN_HEIGHT

        return width, height

    def get_current_state(self) -> tuple[bytes, str]:
        page = self.browser_controller.context.pages[-1]  # Get the latest active page
        screenshot_bytes = page.screenshot(type="png")
        return screenshot_bytes, page.url

    def run_loop(self):
        try:
            while True:
                # Add the current screenshot to the contents for the next model call
                screenshot_bytes, url = self.get_current_state()
                # First prompt includes the system instructions and current URL
                if not self.contents:
                    system_prompt = (
                        "You are a student who's aim is to complete the Huawei course. "
                        "Your task is to complete the course materials and quizzes, complete the quizzes, and scroll through the course materials, DO NOT watch the videos, skip until the end of them. "
                        "Finish until each part has a green checkmark. "
                        "Use only these available custom tools: click_at, multiple_clicks, click_material_until_completed, hover_at, scroll_document, scroll_at, wait_x_seconds. "
                        f"The current URL of the course page is: {url}. "
                        "Steps:"
                        "1. Start by going through the sections in order, from top to bottom. Expanding it, then clicking on the parts inside. Each section may contain videos, quizzes, or reading materials. "
                        "2. For quizzes, click on the quiz, answer all the questions (you can use external resources to find the answers), and submit the quiz. Make sure to get a full score on the quiz. "
                        "3. For reading materials, use the click_material_until_completed tool, provide it with the necessary parameters. "
                        "4. For videos, skip to the end of the video without watching it. You can do this by clicking on the progress bar at the bottom of the video player. Then wait a bit until it shows completed. "
                        "   - If it still doesn't show completed, try clicking the progress bar again, or try scrolling the video player a bit to trigger any lazy loading. "
                        "Notes: "
                        "- You might have to open up sub-sections to view un-completed parts of the course. Make sure to scroll through all the materials in each section. "
                        "- The sections are usually on the left side of the page, and you can click to open them. "
                        "- Uncompleted parts have an empty progress circle next to them, though you have to open up the sections to see them. "
                        "- If there is an arrow pointing to the right at a section, that means its closed, if the arrow points down, its open. "
                        "- Click on the center of the text of the section to open it. Or on the center of the text of a part to complete it. "
                        "- DO NOT click on the buttons at the bottom of the page that say 'Previous' and 'Next', those are for navigating between main sections of the course, not for completing the materials. "
                    )
                    self.contents.append(
                        Content(
                            role="user",
                            parts=[
                                Part(text=system_prompt.strip()),
                                Part.from_bytes(
                                    data=screenshot_bytes, mime_type="image/png"
                                ),
                            ],
                        )
                    )
                # Switch to new tab on final exam
                if (
                    "iexam" in self.browser_controller.context.pages[-1].url
                    and url != self.browser_controller.context.pages[-1].url
                ):
                    print(
                        "Switched to new tab:",
                        self.browser_controller.context.pages[-1].url,
                    )
                    self.browser_controller.context.pages[-1].bring_to_front()
                    screenshot_bytes, url = self.get_current_state()

                context_token_count = self._get_context_token_count()
                if context_token_count > CONTEXT_TOKEN_LIMIT:
                    print(
                        "Context exceeded token limit "
                        f"({context_token_count} > {CONTEXT_TOKEN_LIMIT}). "
                        "Clearing conversation history."
                    )
                    self.contents.clear()
                    continue

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=self.contents,  # ty:ignore[invalid-argument-type]
                    config=self.generate_content_config,
                )

                candidate = response.candidates[0]  # ty:ignore[not-subscriptable]
                self.contents.append(candidate.content)  # type: ignore

                has_function_calls = any(part.function_call for part in candidate.content.parts)  # type: ignore
                if not has_function_calls:
                    text_response = " ".join(
                        [part.text for part in candidate.content.parts if part.text]  # type: ignore
                    )
                    print("Agent finished:", text_response)
                    break

                # Execute the function calls in the candidate content
                results = self.execute_functions(candidate)
                if any(result.get("terminated_by_user") for _, result in results):
                    print(
                        "Agent stopped: action denied by user during safety "
                        "confirmation."
                    )
                    break
                # Get function responses (e.g. screenshots after actions) and add them to contents
                function_responses = self.get_function_responses(results)
                self.contents.append(
                    Content(
                        role="user",
                        parts=[Part(function_response=fr) for fr in function_responses],
                    )
                )
        except Exception as e:
            print(f"Error in agent loop: {e}")
        finally:
            self.browser_controller.cleanup()

    def execute_functions(self, candidate):
        page = self.browser_controller.context.pages[-1]  # Get the latest active page
        results: list[tuple[str, dict[str, Any]]] = []

        viewport_width, viewport_height = self._get_viewport_size(page)

        function_calls = []
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

        for function_call in function_calls:
            print(
                f"Executing function call: {function_call.name} with args {function_call.args}"
            )
            action_result = {}
            fname = function_call.name
            args = dict(function_call.args or {})
            print(f"  -> Executing: {fname}")

            try:
                safety_decision = args.get("safety_decision")
                if (
                    isinstance(safety_decision, dict)
                    and safety_decision.get("decision") == "require_confirmation"
                ):
                    explanation = safety_decision.get(
                        "explanation",
                        "The model requested explicit user confirmation.",
                    )
                    print(f"Confirmation required for {fname}: {explanation}")
                    decision = ""
                    while decision.lower() not in ("y", "yes", "n", "no"):
                        decision = input("Proceed with this action? [y/n]: ").strip()

                    if decision.lower() in ("n", "no"):
                        action_result = {
                            "terminated_by_user": True,
                            "safety_decision": safety_decision,
                        }
                        results.append((fname, action_result))
                        break

                    action_result["safety_acknowledgement"] = True

                if fname == "click_at":
                    actual_x = denormalize_x(args["x"], viewport_width)
                    actual_y = denormalize_y(args["y"], viewport_height)
                    self._show_click_marker(page, actual_x, actual_y)
                    time.sleep(0.1)
                    page.mouse.click(actual_x, actual_y)
                elif fname == "multiple_clicks":
                    coordinates = args.get("coordinates", [])
                    if not isinstance(coordinates, list):
                        raise ValueError("'coordinates' must be a list of dicts.")

                    clicked_points: list[dict[str, int]] = []
                    for point in coordinates:
                        if (
                            not isinstance(point, dict)
                            or "x" not in point
                            or "y" not in point
                        ):
                            raise ValueError(
                                "Each coordinate must be a dict containing 'x' and 'y'."
                            )

                        actual_x = denormalize_x(int(point["x"]), viewport_width)
                        actual_y = denormalize_y(int(point["y"]), viewport_height)
                        self._show_click_marker(page, actual_x, actual_y)
                        time.sleep(0.08)
                        page.mouse.click(actual_x, actual_y)
                        clicked_points.append({"x": actual_x, "y": actual_y})
                        time.sleep(0.8)

                    action_result = {
                        "clicked": len(clicked_points),
                        "coordinates": clicked_points,
                    }
                elif fname == "click_material_until_completed":
                    actual_x = denormalize_x(args["x"], viewport_width)
                    actual_y = denormalize_y(args["y"], viewport_height)
                    clicks = max(0, int(args.get("clicks", 0)))

                    for _ in range(clicks):
                        self._show_click_marker(page, actual_x, actual_y)
                        time.sleep(0.08)
                        page.mouse.click(actual_x, actual_y)
                        time.sleep(0.25)

                    action_result = {
                        "x": actual_x,
                        "y": actual_y,
                        "clicks": clicks,
                    }
                elif fname == "hover_at":
                    actual_x = denormalize_x(args["x"], viewport_width)
                    actual_y = denormalize_y(args["y"], viewport_height)
                    page.mouse.move(actual_x, actual_y)
                elif fname == "scroll_document":
                    if args["direction"] == "down":
                        page.evaluate("window.scrollBy(0, window.innerHeight);")
                    elif args["direction"] == "up":
                        page.evaluate("window.scrollBy(0, -window.innerHeight);")
                    elif args["direction"] == "left":
                        page.evaluate("window.scrollBy(-window.innerWidth, 0);")
                    elif args["direction"] == "right":
                        page.evaluate("window.scrollBy(window.innerWidth, 0);")
                elif fname == "scroll_at":
                    actual_x = denormalize_x(args["x"], viewport_width)
                    actual_y = denormalize_y(args["y"], viewport_height)
                    direction = args.get("direction", "down")
                    magnitude = abs(int(args.get("magnitude", 800)))

                    delta_x = 0
                    delta_y = 0
                    if direction == "down":
                        delta_y = magnitude
                    elif direction == "up":
                        delta_y = -magnitude
                    elif direction == "left":
                        delta_x = -magnitude
                    elif direction == "right":
                        delta_x = magnitude
                    else:
                        raise ValueError(f"Unsupported scroll direction: {direction}")

                    page.mouse.move(actual_x, actual_y)
                    page.mouse.wheel(delta_x, delta_y)
                    action_result = {
                        "x": actual_x,
                        "y": actual_y,
                        "direction": direction,
                        "magnitude": magnitude,
                    }
                elif fname == "wait_x_seconds":
                    wait_seconds = max(0.0, float(args.get("seconds", 0)))
                    time.sleep(wait_seconds)
                    action_result = {"seconds": wait_seconds}

                else:
                    print(f"Warning: Unimplemented or custom function {fname}")

                # Wait for potential navigations/renders
                page.wait_for_load_state(timeout=5000)
                time.sleep(0.75)

            except Exception as e:
                print(f"Error executing {fname}: {e}")
                action_result = {"error": str(e)}

            results.append((fname, action_result))

        return results

    def get_function_responses(self, results):
        page = self.browser_controller.context.pages[-1]  # Get the latest active page
        screenshot_bytes = page.screenshot(type="png")
        current_url = page.url
        function_responses = []
        for name, result in results:
            response_data = {"url": current_url}
            response_data.update(result)
            function_responses.append(
                types.FunctionResponse(
                    name=name,
                    response=response_data,
                    parts=[
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_bytes
                            )
                        )
                    ],
                )
            )
        return function_responses
