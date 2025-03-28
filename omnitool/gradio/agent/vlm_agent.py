import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage
# Local Qwen Addition
from agent.llm_utils.vllm_qwen_interleaved import run_vllm_qwen_interleaved
#
from agent.llm_utils.oaiclient import run_oai_interleaved
from agent.llm_utils.groqclient import run_groq_interleaved
from agent.llm_utils.utils import is_image_path
import time
import re

OUTPUT_DIR = "./tmp/outputs"

def extract_data(input_string, data_type):
    # Regular expression to extract content starting from '```python' until the end if there are no closing backticks
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # Extract content
    # re.DOTALL allows '.' to match newlines as well
    matches = re.findall(pattern, input_string, re.DOTALL)
    # Return the first match if exists, trimming whitespace and ignoring potential closing backticks
    return matches[0][0].strip() if matches else input_string

class VLMAgent:
    def __init__(
        self,
        model: str, 
        provider: str, 
        api_key: str,
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        if model == "omniparser + gpt-4o":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1":
            self.model = "o1"
        elif model == "omniparser + o3-mini":
            self.model = "o3-mini"
        else:
            raise ValueError(f"Model {model} not supported")
        

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0

        self.system = ''
           
    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        # drop looping actions msg, byte image etc
        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1",
                temperature=0,
            )
            print(f"oai token usage: {token_usage}")
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)  # https://openai.com/api/pricing/
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)  # https://openai.com/api/pricing/
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)  # https://openai.com/api/pricing/
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
            )
            print(f"groq token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            vlm_response, token_usage = run_vllm_qwen_interleaved(
                messages=planner_messages,
                system=system,
                model_name=self.model,
                api_key=self.api_key,
                max_tokens=min(2048, self.max_tokens),
                temperature=0
            )
            print(f"qwen token usage: {token_usage}")
            self.total_token_usage += token_usage
            self.total_cost += 0  # local inference is "free"
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, OmniParser: {latency_omniparser:.2f}s", sender="bot")

        print(f"{vlm_response}")
        
        if self.print_usage:
            print(f"Total token so far: {self.total_token_usage}. Total cost so far: $USD{self.total_cost:.5f}")
        
        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))

                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"] 
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)

                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                print(f"Error parsing: {vlm_response_json}")
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
                    f'<details>'
                    f'  <summary>Parsed Screen elemetns by OmniParser</summary>'
                    f'  <pre>{screen_info}</pre>'
                    f'</details>',
                    sender="bot"
                )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        # construct the response so that anthropicExcutor can execute the tool
        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                                            name='computer', type='tool_use')
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                        input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                                        name='computer', type='tool_use')
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(id=f'toolu_{uuid.uuid4()}',
                                            input={'action': vlm_response_json["Next Action"]},
                                            name='computer', type='tool_use')
            response_content.append(sim_content_block)
        response_message = BetaMessage(id=f'toolu_{uuid.uuid4()}', content=response_content, model='', role='assistant', type='message', stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0))
        return response_message, vlm_response_json

    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a Windows device. You can use a mouse and keyboard to interact with the desktop GUI (no terminal or application menu access). You may be given some history of plan/actions. Carefully consider these, along with the task and screenshot, to decide exactly one new action.

Below is the list of bounding boxes by ID on the screen and their description:
{screen_info}

**You must produce exactly one of these "Next Action" each time**:
- "type": type a string of text into the current focus.
- "left_click": move mouse to Box ID and left-click once.
- "right_click": move mouse to Box ID and right-click once.
- "double_click": move mouse to Box ID and double-click.
- "hover": move mouse to Box ID (no click).
- "scroll_up": scroll the screen up.
- "scroll_down": scroll the screen down if content is out of view.
- "wait": wait 1 second.

**Output format** (always valid JSON enclosed in triple backticks):
```json
{{
  "Reasoning": "...",
  "Next Action": "action_type or None",
  "Box ID": 123,
  "value": "... optional"
}}
```

### Important Requirements
1. **Single action**: Only give **one** action in `"Next Action"`.  
2. **JSON only**: Return your response as **valid JSON** code block. No extra text outside it.  
3. If the action is `"type"`, you must include `"value"` with the text to type. Otherwise, omit `"value"`.  
4. If no Box ID is needed (e.g., scroll_down), either omit `"Box ID"` or set `"Box ID": null`.  
5. When you are finished (i.e., no more actions needed), set `"Next Action": "None"`.  
6. For any booleans or empty fields, use valid JSON (`null`, `true`, `false`) instead of Python `None`, `True`, `False`.  
7. Do **not** insert `\nNext Action` or other partial JSON tokens inside the `"Reasoning"` string. Keep `"Reasoning"` a normal text string.

**Examples**:

```json
{{
  "Reasoning": "The screen shows google search results. I want to click the first link to open Amazon.",
  "Next Action": "left_click",
  "Box ID": 0
}}
```

```json
{{
  "Reasoning": "The screen shows Amazon's front page. I need to type \"Apple watch\" in the search bar.",
  "Next Action": "type",
  "Box ID": 2,
  "value": "Apple watch"
}}
```

```json
{{
  "Reasoning": "I've found the item. There's nothing else to do.",
  "Next Action": "None"
}}
```

Remember:
- **Only valid JSON** in triple backticks. 
- No commentary outside the code block.
- Use `null` for missing values, never Python `None`.
"""

        return main_section


def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content 
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place
    """
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep
    
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                # Remove images from SOM or screenshot as needed
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                # VLM shouldn't use anthropic screenshot tool so shouldn't have these but in case it does, remove as needed
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                # Append fixed content to current message's content list
                new_content.append(cnt)
            msg["content"] = new_content
