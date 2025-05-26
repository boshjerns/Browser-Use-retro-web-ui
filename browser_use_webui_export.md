# Browser Use WebUI - Exported Python Files

Generated on: 2025-05-26 08:24:04

## Table of Contents

1. [agent_settings_tab.py](#agent-settings-tabpy)
2. [browser_settings_tab.py](#browser-settings-tabpy)
3. [browser_use_agent_tab.py](#browser-use-agent-tabpy)
4. [interface.py](#interfacepy)
5. [deep_research_agent_tab.py](#deep-research-agent-tabpy)
6. [load_save_config_tab.py](#load-save-config-tabpy)
7. [webui.py](#webuipy)

---

## agent_settings_tab.py

**Path:** `src/webui/components/agent_settings_tab.py`

**Lines:** 269

```python
import json
import os

import gradio as gr
from gradio.components import Component
from typing import Any, Dict, Optional
from src.webui.webui_manager import WebuiManager
from src.utils import config
import logging
from functools import partial

logger = logging.getLogger(__name__)


def update_model_dropdown(llm_provider):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use predefined models for the selected provider
    if llm_provider in config.model_names:
        return gr.Dropdown(choices=config.model_names[llm_provider], value=config.model_names[llm_provider][0],
                           interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server.
    """
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    if not mcp_file or not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)


def create_agent_settings_tab(webui_manager: WebuiManager):
    """
    Creates an agent settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Column():
            override_system_prompt = gr.Textbox(label="Override system prompt", lines=4, interactive=True)
            extend_system_prompt = gr.Textbox(label="Extend system prompt", lines=4, interactive=True)

    with gr.Group():
        mcp_json_file = gr.File(label="MCP server json", interactive=True, file_types=[".json"])
        mcp_server_config = gr.Textbox(label="MCP server", lines=6, interactive=True, visible=False)

    with gr.Group():
        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="LLM Provider",
                value=os.getenv("DEFAULT_LLM", "openai"),
                info="Select LLM provider for LLM",
                interactive=True
            )
            llm_model_name = gr.Dropdown(
                label="LLM Model Name",
                choices=config.model_names[os.getenv("DEFAULT_LLM", "openai")],
                value=config.model_names[os.getenv("DEFAULT_LLM", "openai")][0],
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            use_vision = gr.Checkbox(
                label="Use Vision",
                value=True,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            llm_base_url = gr.Textbox(
                label="Base URL",
                value="",
                info="API endpoint URL (if required)"
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            planner_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Planner LLM Provider",
                info="Select LLM provider for LLM",
                value=None,
                interactive=True
            )
            planner_llm_model_name = gr.Dropdown(
                label="Planner LLM Model Name",
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            planner_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.6,
                step=0.1,
                label="Planner LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            planner_use_vision = gr.Checkbox(
                label="Use Vision(Planner LLM)",
                value=False,
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            planner_ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=16000,
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            planner_llm_base_url = gr.Textbox(
                label="Base URL",
                value="",
                info="API endpoint URL (if required)"
            )
            planner_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value="",
                info="Your API key (leave blank to use .env)"
            )

    with gr.Row():
        max_steps = gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Max Run Steps",
            info="Maximum number of steps the agent will take",
            interactive=True
        )
        max_actions = gr.Slider(
            minimum=1,
            maximum=100,
            value=10,
            step=1,
            label="Max Number of Actions",
            info="Maximum number of actions the agent will take per step",
            interactive=True
        )

    with gr.Row():
        max_input_tokens = gr.Number(
            label="Max Input Tokens",
            value=128000,
            precision=0,
            interactive=True
        )
        tool_calling_method = gr.Dropdown(
            label="Tool Calling Method",
            value="auto",
            interactive=True,
            allow_custom_value=True,
            choices=['function_calling', 'json_mode', 'raw', 'auto', 'tools', "None"],
            visible=True
        )
    tab_components.update(dict(
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        use_vision=use_vision,
        ollama_num_ctx=ollama_num_ctx,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        planner_llm_provider=planner_llm_provider,
        planner_llm_model_name=planner_llm_model_name,
        planner_llm_temperature=planner_llm_temperature,
        planner_use_vision=planner_use_vision,
        planner_ollama_num_ctx=planner_ollama_num_ctx,
        planner_llm_base_url=planner_llm_base_url,
        planner_llm_api_key=planner_llm_api_key,
        max_steps=max_steps,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    webui_manager.add_components("agent_settings", tab_components)

    llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=llm_provider,
        outputs=ollama_num_ctx
    )
    llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[llm_provider],
        outputs=[llm_model_name]
    )
    planner_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[planner_llm_provider],
        outputs=[planner_ollama_num_ctx]
    )
    planner_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[planner_llm_provider],
        outputs=[planner_llm_model_name]
    )

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        update_dict = await update_mcp_server(mcp_file, webui_manager)
        yield update_dict

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )

```

---

## browser_settings_tab.py

**Path:** `src/webui/components/browser_settings_tab.py`

**Lines:** 161

```python
import os
from distutils.util import strtobool
import gradio as gr
import logging
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config

logger = logging.getLogger(__name__)

async def close_browser(webui_manager: WebuiManager):
    """
    Close browser
    """
    if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        webui_manager.bu_current_task.cancel()
        webui_manager.bu_current_task = None

    if webui_manager.bu_browser_context:
        logger.info("⚠️ Closing browser context when changing browser config.")
        await webui_manager.bu_browser_context.close()
        webui_manager.bu_browser_context = None

    if webui_manager.bu_browser:
        logger.info("⚠️ Closing browser when changing browser config.")
        await webui_manager.bu_browser.close()
        webui_manager.bu_browser = None

def create_browser_settings_tab(webui_manager: WebuiManager):
    """
    Creates a browser settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Row():
            browser_binary_path = gr.Textbox(
                label="Browser Binary Path",
                lines=1,
                interactive=True,
                placeholder="e.g. '/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome'"
            )
            browser_user_data_dir = gr.Textbox(
                label="Browser User Data Dir",
                lines=1,
                interactive=True,
                placeholder="Leave it empty if you use your default user data",
            )
    with gr.Group():
        with gr.Row():
            use_own_browser = gr.Checkbox(
                label="Use Own Browser",
                value=bool(strtobool(os.getenv("USE_OWN_BROWSER", "false"))),
                info="Use your existing browser instance",
                interactive=True
            )
            keep_browser_open = gr.Checkbox(
                label="Keep Browser Open",
                value=bool(strtobool(os.getenv("KEEP_BROWSER_OPEN", "true"))),
                info="Keep Browser Open between Tasks",
                interactive=True
            )
            headless = gr.Checkbox(
                label="Headless Mode",
                value=False,
                info="Run browser without GUI",
                interactive=True
            )
            disable_security = gr.Checkbox(
                label="Disable Security",
                value=False,
                info="Disable browser security",
                interactive=True
            )

    with gr.Group():
        with gr.Row():
            window_w = gr.Number(
                label="Window Width",
                value=1280,
                info="Browser window width",
                interactive=True
            )
            window_h = gr.Number(
                label="Window Height",
                value=1100,
                info="Browser window height",
                interactive=True
            )
    with gr.Group():
        with gr.Row():
            cdp_url = gr.Textbox(
                label="CDP URL",
                value=os.getenv("BROWSER_CDP", None),
                info="CDP URL for browser remote debugging",
                interactive=True,
            )
            wss_url = gr.Textbox(
                label="WSS URL",
                info="WSS URL for browser remote debugging",
                interactive=True,
            )
    with gr.Group():
        with gr.Row():
            save_recording_path = gr.Textbox(
                label="Recording Path",
                placeholder="e.g. ./tmp/record_videos",
                info="Path to save browser recordings",
                interactive=True,
            )

            save_trace_path = gr.Textbox(
                label="Trace Path",
                placeholder="e.g. ./tmp/traces",
                info="Path to save Agent traces",
                interactive=True,
            )

        with gr.Row():
            save_agent_history_path = gr.Textbox(
                label="Agent History Save Path",
                value="./tmp/agent_history",
                info="Specify the directory where agent history should be saved.",
                interactive=True,
            )
            save_download_path = gr.Textbox(
                label="Save Directory for browser downloads",
                value="./tmp/downloads",
                info="Specify the directory where downloaded files should be saved.",
                interactive=True,
            )
    tab_components.update(
        dict(
            browser_binary_path=browser_binary_path,
            browser_user_data_dir=browser_user_data_dir,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            save_recording_path=save_recording_path,
            save_trace_path=save_trace_path,
            save_agent_history_path=save_agent_history_path,
            save_download_path=save_download_path,
            cdp_url=cdp_url,
            wss_url=wss_url,
            window_h=window_h,
            window_w=window_w,
        )
    )
    webui_manager.add_components("browser_settings", tab_components)

    async def close_wrapper():
        """Wrapper for handle_clear."""
        await close_browser(webui_manager)

    headless.change(close_wrapper)
    keep_browser_open.change(close_wrapper)
    disable_security.change(close_wrapper)
    use_own_browser.change(close_wrapper)

```

---

## browser_use_agent_tab.py

**Path:** `src/webui/components/browser_use_agent_tab.py`

**Lines:** 1083

```python
import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import gradio as gr

# from browser_use.agent.service import Agent
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.views import BrowserState
from gradio.components import Component
from langchain_core.language_models.chat_models import BaseChatModel

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils import llm_provider
from src.webui.webui_manager import WebuiManager

logger = logging.getLogger(__name__)


# --- Helper Functions --- (Defined at module level)


async def _initialize_llm(
        provider: Optional[str],
        model_name: Optional[str],
        temperature: float,
        base_url: Optional[str],
        api_key: Optional[str],
        num_ctx: Optional[int] = None,
) -> Optional[BaseChatModel]:
    """Initializes the LLM based on settings. Returns None if provider/model is missing."""
    if not provider or not model_name:
        logger.info("LLM Provider or Model Name not specified, LLM will be None.")
        return None
    try:
        # Use your actual LLM provider logic here
        logger.info(
            f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}"
        )
        # Example using a placeholder function
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or None,
            # Add other relevant params like num_ctx for ollama
            num_ctx=num_ctx if provider == "ollama" else None,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        gr.Warning(
            f"Failed to initialize LLM '{model_name}' for provider '{provider}'. Please check settings. Error: {e}"
        )
        return None


def _get_config_value(
        webui_manager: WebuiManager,
        comp_dict: Dict[gr.components.Component, Any],
        comp_id_suffix: str,
        default: Any = None,
) -> Any:
    """Safely get value from component dictionary using its ID suffix relative to the tab."""
    # Assumes component ID format is "tab_name.comp_name"
    tab_name = "browser_use_agent"  # Hardcode or derive if needed
    comp_id = f"{tab_name}.{comp_id_suffix}"
    # Need to find the component object first using the ID from the manager
    try:
        comp = webui_manager.get_component_by_id(comp_id)
        return comp_dict.get(comp, default)
    except KeyError:
        # Try accessing settings tabs as well
        for prefix in ["agent_settings", "browser_settings"]:
            try:
                comp_id = f"{prefix}.{comp_id_suffix}"
                comp = webui_manager.get_component_by_id(comp_id)
                return comp_dict.get(comp, default)
            except KeyError:
                continue
        logger.warning(
            f"Component with suffix '{comp_id_suffix}' not found in manager for value lookup."
        )
        return default


def _format_agent_output(model_output: AgentOutput) -> str:
    """Formats AgentOutput for display in the chatbot using JSON."""
    content = ""
    if model_output:
        try:
            # Directly use model_dump if actions and current_state are Pydantic models
            action_dump = [
                action.model_dump(exclude_none=True) for action in model_output.action
            ]

            state_dump = model_output.current_state.model_dump(exclude_none=True)
            model_output_dump = {
                "current_state": state_dump,
                "action": action_dump,
            }
            # Dump to JSON string with indentation
            json_string = json.dumps(model_output_dump, indent=4, ensure_ascii=False)
            # Wrap in <pre><code> for proper display in HTML
            content = f"<pre><code class='language-json'>{json_string}</code></pre>"

        except AttributeError as ae:
            logger.error(
                f"AttributeError during model dump: {ae}. Check if 'action' or 'current_state' or their items support 'model_dump'."
            )
            content = f"<pre><code>Error: Could not format agent output (AttributeError: {ae}).\nRaw output: {str(model_output)}</code></pre>"
        except Exception as e:
            logger.error(f"Error formatting agent output: {e}", exc_info=True)
            # Fallback to simple string representation on error
            content = f"<pre><code>Error formatting agent output.\nRaw output:\n{str(model_output)}</code></pre>"

    return content.strip()


# --- Updated Callback Implementation ---


async def _handle_new_step(
        webui_manager: WebuiManager, state: BrowserState, output: AgentOutput, step_num: int
):
    """Callback for each step taken by the agent, including screenshot display."""

    # Use the correct chat history attribute name from the user's code
    if not hasattr(webui_manager, "bu_chat_history"):
        logger.error(
            "Attribute 'bu_chat_history' not found in webui_manager! Cannot add chat message."
        )
        # Initialize it maybe? Or raise an error? For now, log and potentially skip chat update.
        webui_manager.bu_chat_history = []  # Initialize if missing (consider if this is the right place)
        # return # Or stop if this is critical
    step_num -= 1
    logger.info(f"Step {step_num} completed.")

    # --- Screenshot Handling ---
    screenshot_html = ""
    # Ensure state.screenshot exists and is not empty before proceeding
    # Use getattr for safer access
    screenshot_data = getattr(state, "screenshot", None)
    if screenshot_data:
        try:
            # Basic validation: check if it looks like base64
            if (
                    isinstance(screenshot_data, str) and len(screenshot_data) > 100
            ):  # Arbitrary length check
                # *** UPDATED STYLE: Removed centering, adjusted width ***
                img_tag = f'<img src="data:image/jpeg;base64,{screenshot_data}" alt="Step {step_num} Screenshot" style="max-width: 800px; max-height: 600px; object-fit:contain;" />'
                screenshot_html = (
                        img_tag + "<br/>"
                )  # Use <br/> for line break after inline-block image
            else:
                logger.warning(
                    f"Screenshot for step {step_num} seems invalid (type: {type(screenshot_data)}, len: {len(screenshot_data) if isinstance(screenshot_data, str) else 'N/A'})."
                )
                screenshot_html = "**[Invalid screenshot data]**<br/>"

        except Exception as e:
            logger.error(
                f"Error processing or formatting screenshot for step {step_num}: {e}",
                exc_info=True,
            )
            screenshot_html = "**[Error displaying screenshot]**<br/>"
    else:
        logger.debug(f"No screenshot available for step {step_num}.")

    # --- Format Agent Output ---
    formatted_output = _format_agent_output(output)  # Use the updated function

    # --- Combine and Append to Chat ---
    step_header = f"--- **Step {step_num}** ---"
    # Combine header, image (with line break), and JSON block
    final_content = step_header + "<br/>" + screenshot_html + formatted_output

    chat_message = {
        "role": "assistant",
        "content": final_content.strip(),  # Remove leading/trailing whitespace
    }

    # Append to the correct chat history list
    webui_manager.bu_chat_history.append(chat_message)

    await asyncio.sleep(0.05)


def _handle_done(webui_manager: WebuiManager, history: AgentHistoryList):
    """Callback when the agent finishes the task (success or failure)."""
    logger.info(
        f"Agent task finished. Duration: {history.total_duration_seconds():.2f}s, Tokens: {history.total_input_tokens()}"
    )
    final_summary = "**Task Completed**\n"
    final_summary += f"- Duration: {history.total_duration_seconds():.2f} seconds\n"
    final_summary += f"- Total Input Tokens: {history.total_input_tokens()}\n"  # Or total tokens if available

    final_result = history.final_result()
    if final_result:
        final_summary += f"- Final Result: {final_result}\n"

    errors = history.errors()
    if errors and any(errors):
        final_summary += f"- **Errors:**\n```\n{errors}\n```\n"
    else:
        final_summary += "- Status: Success\n"

    webui_manager.bu_chat_history.append(
        {"role": "assistant", "content": final_summary}
    )


async def _ask_assistant_callback(
        webui_manager: WebuiManager, query: str, browser_context: BrowserContext
) -> Dict[str, Any]:
    """Callback triggered by the agent's ask_for_assistant action."""
    logger.info("Agent requires assistance. Waiting for user input.")

    if not hasattr(webui_manager, "_chat_history"):
        logger.error("Chat history not found in webui_manager during ask_assistant!")
        return {"response": "Internal Error: Cannot display help request."}

    webui_manager.bu_chat_history.append(
        {
            "role": "assistant",
            "content": f"**Need Help:** {query}\nPlease provide information or perform the required action in the browser, then type your response/confirmation below and click 'Submit Response'.",
        }
    )

    # Use state stored in webui_manager
    webui_manager.bu_response_event = asyncio.Event()
    webui_manager.bu_user_help_response = None  # Reset previous response

    try:
        logger.info("Waiting for user response event...")
        await asyncio.wait_for(
            webui_manager.bu_response_event.wait(), timeout=3600.0
        )  # Long timeout
        logger.info("User response event received.")
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for user assistance.")
        webui_manager.bu_chat_history.append(
            {
                "role": "assistant",
                "content": "**Timeout:** No response received. Trying to proceed.",
            }
        )
        webui_manager.bu_response_event = None  # Clear the event
        return {"response": "Timeout: User did not respond."}  # Inform the agent

    response = webui_manager.bu_user_help_response
    webui_manager.bu_chat_history.append(
        {"role": "user", "content": response}
    )  # Show user response in chat
    webui_manager.bu_response_event = (
        None  # Clear the event for the next potential request
    )
    return {"response": response}


# --- Core Agent Execution Logic --- (Needs access to webui_manager)


async def run_agent_task(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
) -> AsyncGenerator[Dict[gr.components.Component, Any], None]:
    """Handles the entire lifecycle of initializing and running the agent."""

    # --- Get Components ---
    # Need handles to specific UI components to update them
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    run_button_comp = webui_manager.get_component_by_id("browser_use_agent.run_button")
    stop_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.stop_button"
    )
    pause_resume_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.pause_resume_button"
    )
    clear_button_comp = webui_manager.get_component_by_id(
        "browser_use_agent.clear_button"
    )
    chatbot_comp = webui_manager.get_component_by_id("browser_use_agent.chatbot")
    history_file_comp = webui_manager.get_component_by_id(
        "browser_use_agent.agent_history_file"
    )
    gif_comp = webui_manager.get_component_by_id("browser_use_agent.recording_gif")
    browser_view_comp = webui_manager.get_component_by_id(
        "browser_use_agent.browser_view"
    )

    # --- 1. Get Task and Initial UI Update ---
    task = components.get(user_input_comp, "").strip()
    if not task:
        gr.Warning("Please enter a task.")
        yield {run_button_comp: gr.update(interactive=True)}
        return

    # Set running state indirectly via _current_task
    webui_manager.bu_chat_history.append({"role": "user", "content": task})

    yield {
        user_input_comp: gr.Textbox(
            value="", interactive=False, placeholder="Agent is running..."
        ),
        run_button_comp: gr.Button(value="⏳ Running...", interactive=False),
        stop_button_comp: gr.Button(interactive=True),
        pause_resume_button_comp: gr.Button(value="⏸️ Pause", interactive=True),
        clear_button_comp: gr.Button(interactive=False),
        chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
        history_file_comp: gr.update(value=None),
        gif_comp: gr.update(value=None),
    }

    # --- Agent Settings ---
    # Access settings values via components dict, getting IDs from webui_manager
    def get_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"agent_settings.{key}")
        return components.get(comp, default) if comp else default

    override_system_prompt = get_setting("override_system_prompt") or None
    extend_system_prompt = get_setting("extend_system_prompt") or None
    llm_provider_name = get_setting(
        "llm_provider", None
    )  # Default to None if not found
    llm_model_name = get_setting("llm_model_name", None)
    llm_temperature = get_setting("llm_temperature", 0.6)
    use_vision = get_setting("use_vision", True)
    ollama_num_ctx = get_setting("ollama_num_ctx", 16000)
    llm_base_url = get_setting("llm_base_url") or None
    llm_api_key = get_setting("llm_api_key") or None
    max_steps = get_setting("max_steps", 100)
    max_actions = get_setting("max_actions", 10)
    max_input_tokens = get_setting("max_input_tokens", 128000)
    tool_calling_str = get_setting("tool_calling_method", "auto")
    tool_calling_method = tool_calling_str if tool_calling_str != "None" else None
    mcp_server_config_comp = webui_manager.id_to_component.get(
        "agent_settings.mcp_server_config"
    )
    mcp_server_config_str = (
        components.get(mcp_server_config_comp) if mcp_server_config_comp else None
    )
    mcp_server_config = (
        json.loads(mcp_server_config_str) if mcp_server_config_str else None
    )

    # Planner LLM Settings (Optional)
    planner_llm_provider_name = get_setting("planner_llm_provider") or None
    planner_llm = None
    planner_use_vision = False
    if planner_llm_provider_name:
        planner_llm_model_name = get_setting("planner_llm_model_name")
        planner_llm_temperature = get_setting("planner_llm_temperature", 0.6)
        planner_ollama_num_ctx = get_setting("planner_ollama_num_ctx", 16000)
        planner_llm_base_url = get_setting("planner_llm_base_url") or None
        planner_llm_api_key = get_setting("planner_llm_api_key") or None
        planner_use_vision = get_setting("planner_use_vision", False)

        planner_llm = await _initialize_llm(
            planner_llm_provider_name,
            planner_llm_model_name,
            planner_llm_temperature,
            planner_llm_base_url,
            planner_llm_api_key,
            planner_ollama_num_ctx if planner_llm_provider_name == "ollama" else None,
        )

    # --- Browser Settings ---
    def get_browser_setting(key, default=None):
        comp = webui_manager.id_to_component.get(f"browser_settings.{key}")
        return components.get(comp, default) if comp else default

    browser_binary_path = get_browser_setting("browser_binary_path") or None
    browser_user_data_dir = get_browser_setting("browser_user_data_dir") or None
    use_own_browser = get_browser_setting(
        "use_own_browser", False
    )  # Logic handled by CDP/WSS presence
    keep_browser_open = get_browser_setting("keep_browser_open", False)
    headless = get_browser_setting("headless", False)
    disable_security = get_browser_setting("disable_security", False)
    window_w = int(get_browser_setting("window_w", 1280))
    window_h = int(get_browser_setting("window_h", 1100))
    cdp_url = get_browser_setting("cdp_url") or None
    wss_url = get_browser_setting("wss_url") or None
    save_recording_path = get_browser_setting("save_recording_path") or None
    save_trace_path = get_browser_setting("save_trace_path") or None
    save_agent_history_path = get_browser_setting(
        "save_agent_history_path", "./tmp/agent_history"
    )
    save_download_path = get_browser_setting("save_download_path", "./tmp/downloads")

    stream_vw = 70
    stream_vh = int(70 * window_h // window_w)

    os.makedirs(save_agent_history_path, exist_ok=True)
    if save_recording_path:
        os.makedirs(save_recording_path, exist_ok=True)
    if save_trace_path:
        os.makedirs(save_trace_path, exist_ok=True)
    if save_download_path:
        os.makedirs(save_download_path, exist_ok=True)

    # --- 2. Initialize LLM ---
    main_llm = await _initialize_llm(
        llm_provider_name,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        ollama_num_ctx if llm_provider_name == "ollama" else None,
    )

    # Pass the webui_manager instance to the callback when wrapping it
    async def ask_callback_wrapper(
            query: str, browser_context: BrowserContext
    ) -> Dict[str, Any]:
        return await _ask_assistant_callback(webui_manager, query, browser_context)

    if not webui_manager.bu_controller:
        webui_manager.bu_controller = CustomController(
            ask_assistant_callback=ask_callback_wrapper
        )
        await webui_manager.bu_controller.setup_mcp_client(mcp_server_config)

    # --- 4. Initialize Browser and Context ---
    should_close_browser_on_finish = not keep_browser_open

    try:
        # Close existing resources if not keeping open
        if not keep_browser_open:
            if webui_manager.bu_browser_context:
                logger.info("Closing previous browser context.")
                await webui_manager.bu_browser_context.close()
                webui_manager.bu_browser_context = None
            if webui_manager.bu_browser:
                logger.info("Closing previous browser.")
                await webui_manager.bu_browser.close()
                webui_manager.bu_browser = None

        # Create Browser if needed
        if not webui_manager.bu_browser:
            logger.info("Launching new browser instance.")
            extra_args = []
            if use_own_browser:
                browser_binary_path = os.getenv("BROWSER_PATH", None) or browser_binary_path
                if browser_binary_path == "":
                    browser_binary_path = None
                browser_user_data = browser_user_data_dir or os.getenv("BROWSER_USER_DATA", None)
                if browser_user_data:
                    extra_args += [f"--user-data-dir={browser_user_data}"]
            else:
                browser_binary_path = None

            webui_manager.bu_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    browser_binary_path=browser_binary_path,
                    extra_browser_args=extra_args,
                    wss_url=wss_url,
                    cdp_url=cdp_url,
                    new_context_config=BrowserContextConfig(
                        window_width=window_w,
                        window_height=window_h,
                    )
                )
            )

        # Create Context if needed
        if not webui_manager.bu_browser_context:
            logger.info("Creating new browser context.")
            context_config = BrowserContextConfig(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path
                if save_recording_path
                else None,
                save_downloads_path=save_download_path if save_download_path else None,
                window_height=window_h,
                window_width=window_w,
            )
            if not webui_manager.bu_browser:
                raise ValueError("Browser not initialized, cannot create context.")
            webui_manager.bu_browser_context = (
                await webui_manager.bu_browser.new_context(config=context_config)
            )

        # --- 5. Initialize or Update Agent ---
        webui_manager.bu_agent_task_id = str(uuid.uuid4())  # New ID for this task run
        os.makedirs(
            os.path.join(save_agent_history_path, webui_manager.bu_agent_task_id),
            exist_ok=True,
        )
        history_file = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.json",
        )
        gif_path = os.path.join(
            save_agent_history_path,
            webui_manager.bu_agent_task_id,
            f"{webui_manager.bu_agent_task_id}.gif",
        )

        # Pass the webui_manager to callbacks when wrapping them
        async def step_callback_wrapper(
                state: BrowserState, output: AgentOutput, step_num: int
        ):
            await _handle_new_step(webui_manager, state, output, step_num)

        def done_callback_wrapper(history: AgentHistoryList):
            _handle_done(webui_manager, history)

        if not webui_manager.bu_agent:
            logger.info(f"Initializing new agent for task: {task}")
            if not webui_manager.bu_browser or not webui_manager.bu_browser_context:
                raise ValueError(
                    "Browser or Context not initialized, cannot create agent."
                )
            webui_manager.bu_agent = BrowserUseAgent(
                task=task,
                llm=main_llm,
                browser=webui_manager.bu_browser,
                browser_context=webui_manager.bu_browser_context,
                controller=webui_manager.bu_controller,
                register_new_step_callback=step_callback_wrapper,
                register_done_callback=done_callback_wrapper,
                use_vision=use_vision,
                override_system_message=override_system_prompt,
                extend_system_message=extend_system_prompt,
                max_input_tokens=max_input_tokens,
                max_actions_per_step=max_actions,
                tool_calling_method=tool_calling_method,
                planner_llm=planner_llm,
                use_vision_for_planner=planner_use_vision if planner_llm else False,
                source="webui",
            )
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.settings.generate_gif = gif_path
        else:
            webui_manager.bu_agent.state.agent_id = webui_manager.bu_agent_task_id
            webui_manager.bu_agent.add_new_task(task)
            webui_manager.bu_agent.settings.generate_gif = gif_path
            webui_manager.bu_agent.browser = webui_manager.bu_browser
            webui_manager.bu_agent.browser_context = webui_manager.bu_browser_context
            webui_manager.bu_agent.controller = webui_manager.bu_controller

        # --- 6. Run Agent Task and Stream Updates ---
        agent_run_coro = webui_manager.bu_agent.run(max_steps=max_steps)
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.bu_current_task = agent_task  # Store the task

        last_chat_len = len(webui_manager.bu_chat_history)
        while not agent_task.done():
            is_paused = webui_manager.bu_agent.state.paused
            is_stopped = webui_manager.bu_agent.state.stopped

            # Check for pause state
            if is_paused:
                yield {
                    pause_resume_button_comp: gr.update(
                        value="▶️ Resume", interactive=True
                    ),
                    stop_button_comp: gr.update(interactive=True),
                }
                # Wait until pause is released or task is stopped/done
                while is_paused and not agent_task.done():
                    # Re-check agent state in loop
                    is_paused = webui_manager.bu_agent.state.paused
                    is_stopped = webui_manager.bu_agent.state.stopped
                    if is_stopped:  # Stop signal received while paused
                        break
                    await asyncio.sleep(0.2)

                if (
                        agent_task.done() or is_stopped
                ):  # If stopped or task finished while paused
                    break

                # If resumed, yield UI update
                yield {
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=True
                    ),
                    run_button_comp: gr.update(
                        value="⏳ Running...", interactive=False
                    ),
                }

            # Check if agent stopped itself or stop button was pressed (which sets agent.state.stopped)
            if is_stopped:
                logger.info("Agent has stopped (internally or via stop button).")
                if not agent_task.done():
                    # Ensure the task coroutine finishes if agent just set flag
                    try:
                        await asyncio.wait_for(
                            agent_task, timeout=1.0
                        )  # Give it a moment to exit run()
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Agent task did not finish quickly after stop signal, cancelling."
                        )
                        agent_task.cancel()
                    except Exception:  # Catch task exceptions if it errors on stop
                        pass
                break  # Exit the streaming loop

            # Check if agent is asking for help (via response_event)
            update_dict = {}
            if webui_manager.bu_response_event is not None:
                update_dict = {
                    user_input_comp: gr.update(
                        placeholder="Agent needs help. Enter response and submit.",
                        interactive=True,
                    ),
                    run_button_comp: gr.update(
                        value="✔️ Submit Response", interactive=True
                    ),
                    pause_resume_button_comp: gr.update(interactive=False),
                    stop_button_comp: gr.update(interactive=False),
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
                last_chat_len = len(webui_manager.bu_chat_history)
                yield update_dict
                # Wait until response is submitted or task finishes
                while (
                        webui_manager.bu_response_event is not None
                        and not agent_task.done()
                ):
                    await asyncio.sleep(0.2)
                # Restore UI after response submitted or if task ended unexpectedly
                if not agent_task.done():
                    yield {
                        user_input_comp: gr.update(
                            placeholder="Agent is running...", interactive=False
                        ),
                        run_button_comp: gr.update(
                            value="⏳ Running...", interactive=False
                        ),
                        pause_resume_button_comp: gr.update(interactive=True),
                        stop_button_comp: gr.update(interactive=True),
                    }
                else:
                    break  # Task finished while waiting for response

            # Update Chatbot if new messages arrived via callbacks
            if len(webui_manager.bu_chat_history) > last_chat_len:
                update_dict[chatbot_comp] = gr.update(
                    value=webui_manager.bu_chat_history
                )
                last_chat_len = len(webui_manager.bu_chat_history)

            # Update Browser View
            if headless and webui_manager.bu_browser_context:
                try:
                    screenshot_b64 = (
                        await webui_manager.bu_browser_context.take_screenshot()
                    )
                    if screenshot_b64:
                        html_content = f'<img src="data:image/jpeg;base64,{screenshot_b64}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                        update_dict[browser_view_comp] = gr.update(
                            value=html_content, visible=True
                        )
                except Exception as e:
                    logger.debug(f"Failed to capture screenshot: {e}")
                    update_dict[browser_view_comp] = gr.update(
                        value="<div style='...'>Error loading view...</div>",
                        visible=True,
                    )
            else:
                update_dict[browser_view_comp] = gr.update(visible=False)

            # Yield accumulated updates
            if update_dict:
                yield update_dict

            await asyncio.sleep(0.1)  # Polling interval

        # --- 7. Task Finalization ---
        webui_manager.bu_agent.state.paused = False
        webui_manager.bu_agent.state.stopped = False
        final_update = {}
        try:
            logger.info("Agent task completing...")
            # Await the task ensure completion and catch exceptions if not already caught
            if not agent_task.done():
                await agent_task  # Retrieve result/exception
            elif agent_task.exception():  # Check if task finished with exception
                agent_task.result()  # Raise the exception to be caught below
            logger.info("Agent task completed processing.")

            logger.info(f"Explicitly saving agent history to: {history_file}")
            webui_manager.bu_agent.save_history(history_file)

            if os.path.exists(history_file):
                final_update[history_file_comp] = gr.File(value=history_file)

            if gif_path and os.path.exists(gif_path):
                logger.info(f"GIF found at: {gif_path}")
                final_update[gif_comp] = gr.Image(value=gif_path)

        except asyncio.CancelledError:
            logger.info("Agent task was cancelled.")
            if not any(
                    "Cancelled" in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": "**Task Cancelled**."}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            error_message = (
                f"**Agent Execution Error:**\n```\n{type(e).__name__}: {e}\n```"
            )
            if not any(
                    error_message in msg.get("content", "")
                    for msg in webui_manager.bu_chat_history
                    if msg.get("role") == "assistant"
            ):
                webui_manager.bu_chat_history.append(
                    {"role": "assistant", "content": error_message}
                )
            final_update[chatbot_comp] = gr.update(value=webui_manager.bu_chat_history)
            gr.Error(f"Agent execution failed: {e}")

        finally:
            webui_manager.bu_current_task = None  # Clear the task reference

            # Close browser/context if requested
            if should_close_browser_on_finish:
                if webui_manager.bu_browser_context:
                    logger.info("Closing browser context after task.")
                    await webui_manager.bu_browser_context.close()
                    webui_manager.bu_browser_context = None
                if webui_manager.bu_browser:
                    logger.info("Closing browser after task.")
                    await webui_manager.bu_browser.close()
                    webui_manager.bu_browser = None

            # --- 8. Final UI Update ---
            final_update.update(
                {
                    user_input_comp: gr.update(
                        value="",
                        interactive=True,
                        placeholder="Enter your next task...",
                    ),
                    run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
                    stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
                    pause_resume_button_comp: gr.update(
                        value="⏸️ Pause", interactive=False
                    ),
                    clear_button_comp: gr.update(interactive=True),
                    # Ensure final chat history is shown
                    chatbot_comp: gr.update(value=webui_manager.bu_chat_history),
                }
            )
            yield final_update

    except Exception as e:
        # Catch errors during setup (before agent run starts)
        logger.error(f"Error setting up agent task: {e}", exc_info=True)
        webui_manager.bu_current_task = None  # Ensure state is reset
        yield {
            user_input_comp: gr.update(
                interactive=True, placeholder="Error during setup. Enter task..."
            ),
            run_button_comp: gr.update(value="▶️ Submit Task", interactive=True),
            stop_button_comp: gr.update(value="⏹️ Stop", interactive=False),
            pause_resume_button_comp: gr.update(value="⏸️ Pause", interactive=False),
            clear_button_comp: gr.update(interactive=True),
            chatbot_comp: gr.update(
                value=webui_manager.bu_chat_history
                      + [{"role": "assistant", "content": f"**Setup Error:** {e}"}]
            ),
        }


# --- Button Click Handlers --- (Need access to webui_manager)


async def handle_submit(
        webui_manager: WebuiManager, components: Dict[gr.components.Component, Any]
):
    """Handles clicks on the main 'Submit' button."""
    user_input_comp = webui_manager.get_component_by_id("browser_use_agent.user_input")
    user_input_value = components.get(user_input_comp, "").strip()

    # Check if waiting for user assistance
    if webui_manager.bu_response_event and not webui_manager.bu_response_event.is_set():
        logger.info(f"User submitted assistance: {user_input_value}")
        webui_manager.bu_user_help_response = (
            user_input_value if user_input_value else "User provided no text response."
        )
        webui_manager.bu_response_event.set()
        # UI updates handled by the main loop reacting to the event being set
        yield {
            user_input_comp: gr.update(
                value="",
                interactive=False,
                placeholder="Waiting for agent to continue...",
            ),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(value="⏳ Running...", interactive=False),
        }
    # Check if a task is currently running (using _current_task)
    elif webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        logger.warning(
            "Submit button clicked while agent is already running and not asking for help."
        )
        gr.Info("Agent is currently running. Please wait or use Stop/Pause.")
        yield {}  # No change
    else:
        # Handle submission for a new task
        logger.info("Submit button clicked for new task.")
        # Use async generator to stream updates from run_agent_task
        async for update in run_agent_task(webui_manager, components):
            yield update


async def handle_stop(webui_manager: WebuiManager):
    """Handles clicks on the 'Stop' button."""
    logger.info("Stop button clicked.")
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        # Signal the agent to stop by setting its internal flag
        agent.state.stopped = True
        agent.state.paused = False  # Ensure not paused if stopped
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False, value="⏹️ Stopping..."),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=False),
        }
    else:
        logger.warning("Stop clicked but agent is not running or task is already done.")
        # Reset UI just in case it's stuck
        return {
            webui_manager.get_component_by_id(
                "browser_use_agent.run_button"
            ): gr.update(interactive=True),
            webui_manager.get_component_by_id(
                "browser_use_agent.stop_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.pause_resume_button"
            ): gr.update(interactive=False),
            webui_manager.get_component_by_id(
                "browser_use_agent.clear_button"
            ): gr.update(interactive=True),
        }


async def handle_pause_resume(webui_manager: WebuiManager):
    """Handles clicks on the 'Pause/Resume' button."""
    agent = webui_manager.bu_agent
    task = webui_manager.bu_current_task

    if agent and task and not task.done():
        if agent.state.paused:
            logger.info("Resume button clicked.")
            agent.resume()
            # UI update happens in main loop
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="⏸️ Pause", interactive=True)
            }  # Optimistic update
        else:
            logger.info("Pause button clicked.")
            agent.pause()
            return {
                webui_manager.get_component_by_id(
                    "browser_use_agent.pause_resume_button"
                ): gr.update(value="▶️ Resume", interactive=True)
            }  # Optimistic update
    else:
        logger.warning(
            "Pause/Resume clicked but agent is not running or doesn't support state."
        )
        return {}  # No change


async def handle_clear(webui_manager: WebuiManager):
    """Handles clicks on the 'Clear' button."""
    logger.info("Clear button clicked.")

    # Stop any running task first
    task = webui_manager.bu_current_task
    if task and not task.done():
        logger.info("Clearing requires stopping the current task.")
        webui_manager.bu_agent.stop()
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)  # Wait briefly
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error stopping task on clear: {e}")
    webui_manager.bu_current_task = None

    if webui_manager.bu_controller:
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None
    webui_manager.bu_agent = None

    # Reset state stored in manager
    webui_manager.bu_chat_history = []
    webui_manager.bu_response_event = None
    webui_manager.bu_user_help_response = None
    webui_manager.bu_agent_task_id = None

    logger.info("Agent state and browser resources cleared.")

    # Reset UI components
    return {
        webui_manager.get_component_by_id("browser_use_agent.chatbot"): gr.update(
            value=[]
        ),
        webui_manager.get_component_by_id("browser_use_agent.user_input"): gr.update(
            value="", placeholder="Enter your task here..."
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.agent_history_file"
        ): gr.update(value=None),
        webui_manager.get_component_by_id("browser_use_agent.recording_gif"): gr.update(
            value=None
        ),
        webui_manager.get_component_by_id("browser_use_agent.browser_view"): gr.update(
            value="<div style='...'>Browser Cleared</div>"
        ),
        webui_manager.get_component_by_id("browser_use_agent.run_button"): gr.update(
            value="▶️ Submit Task", interactive=True
        ),
        webui_manager.get_component_by_id("browser_use_agent.stop_button"): gr.update(
            interactive=False
        ),
        webui_manager.get_component_by_id(
            "browser_use_agent.pause_resume_button"
        ): gr.update(value="⏸️ Pause", interactive=False),
        webui_manager.get_component_by_id("browser_use_agent.clear_button"): gr.update(
            interactive=True
        ),
    }


# --- Tab Creation Function ---


def create_browser_use_agent_tab(webui_manager: WebuiManager):
    """
    Create the run agent tab, defining UI, state, and handlers.
    """
    webui_manager.init_browser_use_agent()

    # --- Define UI Components ---
    tab_components = {}
    with gr.Column():
        chatbot = gr.Chatbot(
            lambda: webui_manager.bu_chat_history,  # Load history dynamically
            elem_id="browser_use_chatbot",
            label="Agent Interaction",
            type="messages",
            height=600,
            show_copy_button=True,
        )
        user_input = gr.Textbox(
            label="Your Task or Response",
            placeholder="Enter your task here or provide assistance when asked.",
            lines=3,
            interactive=True,
            elem_id="user_input",
        )
        with gr.Row():
            stop_button = gr.Button(
                "⏹️ Stop", interactive=False, variant="stop", scale=2
            )
            pause_resume_button = gr.Button(
                "⏸️ Pause", interactive=False, variant="secondary", scale=2, visible=True
            )
            clear_button = gr.Button(
                "🗑️ Clear", interactive=True, variant="secondary", scale=2
            )
            run_button = gr.Button("▶️ Submit Task", variant="primary", scale=3)

        browser_view = gr.HTML(
            value="<div style='width:100%; height:50vh; display:flex; justify-content:center; align-items:center; border:1px solid #ccc; background-color:#f0f0f0;'><p>Browser View (Requires Headless=True)</p></div>",
            label="Browser Live View",
            elem_id="browser_view",
            visible=False,
        )
        with gr.Column():
            gr.Markdown("### Task Outputs")
            agent_history_file = gr.File(label="Agent History JSON", interactive=False)
            recording_gif = gr.Image(
                label="Task Recording GIF",
                format="gif",
                interactive=False,
                type="filepath",
            )

    # --- Store Components in Manager ---
    tab_components.update(
        dict(
            chatbot=chatbot,
            user_input=user_input,
            clear_button=clear_button,
            run_button=run_button,
            stop_button=stop_button,
            pause_resume_button=pause_resume_button,
            agent_history_file=agent_history_file,
            recording_gif=recording_gif,
            browser_view=browser_view,
        )
    )
    webui_manager.add_components(
        "browser_use_agent", tab_components
    )  # Use "browser_use_agent" as tab_name prefix

    all_managed_components = set(
        webui_manager.get_components()
    )  # Get all components known to manager
    run_tab_outputs = list(tab_components.values())

    async def submit_wrapper(
            components_dict: Dict[Component, Any],
    ) -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_submit that yields its results."""
        async for update in handle_submit(webui_manager, components_dict):
            yield update

    async def stop_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_stop."""
        update_dict = await handle_stop(webui_manager)
        yield update_dict

    async def pause_resume_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_pause_resume."""
        update_dict = await handle_pause_resume(webui_manager)
        yield update_dict

    async def clear_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        """Wrapper for handle_clear."""
        update_dict = await handle_clear(webui_manager)
        yield update_dict

    # --- Connect Event Handlers using the Wrappers --
    run_button.click(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs
    )
    user_input.submit(
        fn=submit_wrapper, inputs=all_managed_components, outputs=run_tab_outputs
    )
    stop_button.click(fn=stop_wrapper, inputs=None, outputs=run_tab_outputs)
    pause_resume_button.click(
        fn=pause_resume_wrapper, inputs=None, outputs=run_tab_outputs
    )
    clear_button.click(fn=clear_wrapper, inputs=None, outputs=run_tab_outputs)

```

---

## interface.py

**Path:** `src/webui/interface.py`

**Lines:** 170

```python
import gradio as gr

# Your component imports
from src.webui.webui_manager import WebuiManager
from src.webui.components.agent_settings_tab import create_agent_settings_tab
from src.webui.components.browser_settings_tab import create_browser_settings_tab
from src.webui.components.browser_use_agent_tab import create_browser_use_agent_tab
from src.webui.components.deep_research_agent_tab import create_deep_research_agent_tab
from src.webui.components.load_save_config_tab import create_load_save_config_tab

theme_map = {
    "Windows 98 RGB": gr.themes.Base(),
    "Ocean": gr.themes.Ocean()
}

def create_ui(theme_name="Windows 98 RGB"):
    head_html = '''
<link rel="stylesheet" href="https://unpkg.com/98.css">
<style>
/* Base body + animated background */
body, html {
    margin: 0;
    padding: 0;
    background: linear-gradient(135deg, #000000, #1a1a1a) fixed;
    font-family: "Pixelated MS Sans Serif", Tahoma, monospace !important;
    color: #00ffcc;
    background-size: 400% 400%;
    animation: rgbWave 10s ease infinite;
}

@keyframes rgbWave {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Container styling */
.gradio-container {
    background-color: rgba(0, 0, 0, 0.6) !important;
    padding: 16px;
    border: 2px solid #00ffff;
    border-radius: 6px;
    box-shadow: 0 0 12px #00ffff88, inset 0 0 6px #00ffff55;
}

/* RGB Title Bar */
.title-bar {
    background: linear-gradient(90deg, red, orange, yellow, green, cyan, blue, violet);
    color: black;
    text-align: center;
    font-weight: bold;
    padding: 4px;
    font-size: 16px;
    border: 2px solid #fff;
    animation: glow 5s linear infinite;
}

@keyframes glow {
  0% { filter: hue-rotate(0deg); }
  100% { filter: hue-rotate(360deg); }
}

/* Window Body */
.window-body {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(3px);
    padding: 12px;
    border: 1px solid #00cccc;
    box-shadow: inset 0 0 4px #00ffff55;
    color: #e0ffff;
    font-size: 14px;
}

/* Tabs RGB highlight */
.gr-tabs > div[role="tablist"] {
    background: transparent;
    border-bottom: 2px solid #00ffff;
}
.gr-tabs button {
    background: #111111;
    color: #00ffcc;
    border: 1px solid #00ffff66;
    margin-right: 4px;
    padding: 4px 10px;
    box-shadow: 0 0 4px #00ffff33;
    font-family: "Pixelated MS Sans Serif", Tahoma, monospace !important;
}
.gr-tabs button.selected {
    background: #00ffff33;
    box-shadow: 0 0 8px #00ffff88;
    border-bottom: none;
}

/* Inputs, buttons, dropdowns styled retro-hacker */
input, textarea, select, button {
    background: black;
    color: #00ffcc;
    border: 1px solid #00ffff;
    font-family: "Pixelated MS Sans Serif", Tahoma, monospace !important;
    padding: 5px 10px;
    border-radius: 3px;
    box-shadow: 0 0 6px #00ffff44;
}
input:hover, textarea:hover, select:hover, button:hover {
    background: #111111;
    box-shadow: 0 0 8px #00ffff99;
}

/* Flickering checkbox effect */
input[type="checkbox"] {
    accent-color: #00ffff;
    box-shadow: 0 0 3px #00ffff;
}

/* File + sliders */
input[type="file"], input[type="range"] {
    background-color: transparent;
    border: none;
    color: #00ffff;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 12px;
}
::-webkit-scrollbar-track {
  background: #111;
}
::-webkit-scrollbar-thumb {
  background: linear-gradient(to bottom, #00ffff, #ff00ff);
  border-radius: 6px;
}
</style>
'''

    ui_manager = WebuiManager()

    with gr.Blocks(title="Browser Use WebUI", theme=theme_map[theme_name], head=head_html) as demo:
        with gr.Row():
            gr.Markdown("# 🖥️ Browser Use WebUI — RGB98", elem_classes=["title-bar"])

        with gr.Tabs(elem_classes=["window"]):
            with gr.TabItem("⚙️ Agent Settings"):
                with gr.Column(elem_classes=["window-body"]):
                    create_agent_settings_tab(ui_manager)

            with gr.TabItem("🌐 Browser Settings"):
                with gr.Column(elem_classes=["window-body"]):
                    create_browser_settings_tab(ui_manager)

            with gr.TabItem("🤖 Run Agent"):
                with gr.Column(elem_classes=["window-body"]):
                    create_browser_use_agent_tab(ui_manager)

            with gr.TabItem("🎁 Agent Marketplace"):
                with gr.Column(elem_classes=["window-body"]):
                    gr.Markdown("### Browse agents in style")
                    with gr.Tabs():
                        with gr.TabItem("Deep Research"):
                            create_deep_research_agent_tab(ui_manager)

            with gr.TabItem("📁 Load & Save Config"):
                with gr.Column(elem_classes=["window-body"]):
                    create_load_save_config_tab(ui_manager)

    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch()

```

---

## deep_research_agent_tab.py

**Path:** `src/webui/components/deep_research_agent_tab.py`

**Lines:** 451

```python
import gradio as gr
from gradio.components import Component
from functools import partial

from src.webui.webui_manager import WebuiManager
from src.utils import config
import logging
import os
from typing import Any, Dict, AsyncGenerator, Optional, Tuple, Union
import asyncio
import json
from src.agent.deep_research.deep_research_agent import DeepResearchAgent
from src.utils import llm_provider

logger = logging.getLogger(__name__)


async def _initialize_llm(provider: Optional[str], model_name: Optional[str], temperature: float,
                          base_url: Optional[str], api_key: Optional[str], num_ctx: Optional[int] = None):
    """Initializes the LLM based on settings. Returns None if provider/model is missing."""
    if not provider or not model_name:
        logger.info("LLM Provider or Model Name not specified, LLM will be None.")
        return None
    try:
        logger.info(f"Initializing LLM: Provider={provider}, Model={model_name}, Temp={temperature}")
        # Use your actual LLM provider logic here
        llm = llm_provider.get_llm_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            base_url=base_url or None,
            api_key=api_key or None,
            num_ctx=num_ctx if provider == "ollama" else None
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        gr.Warning(
            f"Failed to initialize LLM '{model_name}' for provider '{provider}'. Please check settings. Error: {e}")
        return None


def _read_file_safe(file_path: str) -> Optional[str]:
    """Safely read a file, returning None if it doesn't exist or on error."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


# --- Deep Research Agent Specific Logic ---

async def run_deep_research(webui_manager: WebuiManager, components: Dict[Component, Any]) -> AsyncGenerator[
    Dict[Component, Any], None]:
    """Handles initializing and running the DeepResearchAgent."""

    # --- Get Components ---
    research_task_comp = webui_manager.get_component_by_id("deep_research_agent.research_task")
    resume_task_id_comp = webui_manager.get_component_by_id("deep_research_agent.resume_task_id")
    parallel_num_comp = webui_manager.get_component_by_id("deep_research_agent.parallel_num")
    save_dir_comp = webui_manager.get_component_by_id(
        "deep_research_agent.max_query")  # Note: component ID seems misnamed in original code
    start_button_comp = webui_manager.get_component_by_id("deep_research_agent.start_button")
    stop_button_comp = webui_manager.get_component_by_id("deep_research_agent.stop_button")
    markdown_display_comp = webui_manager.get_component_by_id("deep_research_agent.markdown_display")
    markdown_download_comp = webui_manager.get_component_by_id("deep_research_agent.markdown_download")
    mcp_server_config_comp = webui_manager.get_component_by_id("deep_research_agent.mcp_server_config")

    # --- 1. Get Task and Settings ---
    task_topic = components.get(research_task_comp, "").strip()
    task_id_to_resume = components.get(resume_task_id_comp, "").strip() or None
    max_parallel_agents = int(components.get(parallel_num_comp, 1))
    base_save_dir = components.get(save_dir_comp, "./tmp/deep_research")
    mcp_server_config_str = components.get(mcp_server_config_comp)
    mcp_config = json.loads(mcp_server_config_str) if mcp_server_config_str else None

    if not task_topic:
        gr.Warning("Please enter a research task.")
        yield {start_button_comp: gr.update(interactive=True)}  # Re-enable start button
        return

    # Store base save dir for stop handler
    webui_manager.dr_save_dir = base_save_dir
    os.makedirs(base_save_dir, exist_ok=True)

    # --- 2. Initial UI Update ---
    yield {
        start_button_comp: gr.update(value="⏳ Running...", interactive=False),
        stop_button_comp: gr.update(interactive=True),
        research_task_comp: gr.update(interactive=False),
        resume_task_id_comp: gr.update(interactive=False),
        parallel_num_comp: gr.update(interactive=False),
        save_dir_comp: gr.update(interactive=False),
        markdown_display_comp: gr.update(value="Starting research..."),
        markdown_download_comp: gr.update(value=None, interactive=False)
    }

    agent_task = None
    running_task_id = None
    plan_file_path = None
    report_file_path = None
    last_plan_content = None
    last_plan_mtime = 0

    try:
        # --- 3. Get LLM and Browser Config from other tabs ---
        # Access settings values via components dict, getting IDs from webui_manager
        def get_setting(tab: str, key: str, default: Any = None):
            comp = webui_manager.id_to_component.get(f"{tab}.{key}")
            return components.get(comp, default) if comp else default

        # LLM Config (from agent_settings tab)
        llm_provider_name = get_setting("agent_settings", "llm_provider")
        llm_model_name = get_setting("agent_settings", "llm_model_name")
        llm_temperature = max(get_setting("agent_settings", "llm_temperature", 0.5), 0.5)
        llm_base_url = get_setting("agent_settings", "llm_base_url")
        llm_api_key = get_setting("agent_settings", "llm_api_key")
        ollama_num_ctx = get_setting("agent_settings", "ollama_num_ctx")

        llm = await _initialize_llm(
            llm_provider_name, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
            ollama_num_ctx if llm_provider_name == "ollama" else None
        )
        if not llm:
            raise ValueError("LLM Initialization failed. Please check Agent Settings.")

        # Browser Config (from browser_settings tab)
        # Note: DeepResearchAgent constructor takes a dict, not full Browser/Context objects
        browser_config_dict = {
            "headless": get_setting("browser_settings", "headless", False),
            "disable_security": get_setting("browser_settings", "disable_security", False),
            "browser_binary_path": get_setting("browser_settings", "browser_binary_path"),
            "user_data_dir": get_setting("browser_settings", "browser_user_data_dir"),
            "window_width": int(get_setting("browser_settings", "window_w", 1280)),
            "window_height": int(get_setting("browser_settings", "window_h", 1100)),
            # Add other relevant fields if DeepResearchAgent accepts them
        }

        # --- 4. Initialize or Get Agent ---
        if not webui_manager.dr_agent:
            webui_manager.dr_agent = DeepResearchAgent(
                llm=llm,
                browser_config=browser_config_dict,
                mcp_server_config=mcp_config
            )
            logger.info("DeepResearchAgent initialized.")

        # --- 5. Start Agent Run ---
        agent_run_coro = webui_manager.dr_agent.run(
            topic=task_topic,
            task_id=task_id_to_resume,
            save_dir=base_save_dir,
            max_parallel_browsers=max_parallel_agents
        )
        agent_task = asyncio.create_task(agent_run_coro)
        webui_manager.dr_current_task = agent_task

        # Wait briefly for the agent to start and potentially create the task ID/folder
        await asyncio.sleep(1.0)

        # Determine the actual task ID being used (agent sets this)
        running_task_id = webui_manager.dr_agent.current_task_id
        if not running_task_id:
            # Agent might not have set it yet, try to get from result later? Risky.
            # Or derive from resume_task_id if provided?
            running_task_id = task_id_to_resume
            if not running_task_id:
                logger.warning("Could not determine running task ID immediately.")
                # We can still monitor, but might miss initial plan if ID needed for path
            else:
                logger.info(f"Assuming task ID based on resume ID: {running_task_id}")
        else:
            logger.info(f"Agent started with Task ID: {running_task_id}")

        webui_manager.dr_task_id = running_task_id  # Store for stop handler

        # --- 6. Monitor Progress via research_plan.md ---
        if running_task_id:
            task_specific_dir = os.path.join(base_save_dir, str(running_task_id))
            plan_file_path = os.path.join(task_specific_dir, "research_plan.md")
            report_file_path = os.path.join(task_specific_dir, "report.md")
            logger.info(f"Monitoring plan file: {plan_file_path}")
        else:
            logger.warning("Cannot monitor plan file: Task ID unknown.")
            plan_file_path = None
        last_plan_content = None
        while not agent_task.done():
            update_dict = {}
            update_dict[resume_task_id_comp] = gr.update(value=running_task_id)
            agent_stopped = getattr(webui_manager.dr_agent, 'stopped', False)
            if agent_stopped:
                logger.info("Stop signal detected from agent state.")
                break  # Exit monitoring loop

            # Check and update research plan display
            if plan_file_path:
                try:
                    current_mtime = os.path.getmtime(plan_file_path) if os.path.exists(plan_file_path) else 0
                    if current_mtime > last_plan_mtime:
                        logger.info(f"Detected change in {plan_file_path}")
                        plan_content = _read_file_safe(plan_file_path)
                        if last_plan_content is None or (
                                plan_content is not None and plan_content != last_plan_content):
                            update_dict[markdown_display_comp] = gr.update(value=plan_content)
                            last_plan_content = plan_content
                            last_plan_mtime = current_mtime
                        elif plan_content is None:
                            # File might have been deleted or became unreadable
                            last_plan_mtime = 0  # Reset to force re-read attempt later
                except Exception as e:
                    logger.warning(f"Error checking/reading plan file {plan_file_path}: {e}")
                    # Avoid continuous logging for the same error
                    await asyncio.sleep(2.0)

            # Yield updates if any
            if update_dict:
                yield update_dict

            await asyncio.sleep(1.0)  # Check file changes every second

        # --- 7. Task Finalization ---
        logger.info("Agent task processing finished. Awaiting final result...")
        final_result_dict = await agent_task  # Get result or raise exception
        logger.info(f"Agent run completed. Result keys: {final_result_dict.keys() if final_result_dict else 'None'}")

        # Try to get task ID from result if not known before
        if not running_task_id and final_result_dict and 'task_id' in final_result_dict:
            running_task_id = final_result_dict['task_id']
            webui_manager.dr_task_id = running_task_id
            task_specific_dir = os.path.join(base_save_dir, str(running_task_id))
            report_file_path = os.path.join(task_specific_dir, "report.md")
            logger.info(f"Task ID confirmed from result: {running_task_id}")

        final_ui_update = {}
        if report_file_path and os.path.exists(report_file_path):
            logger.info(f"Loading final report from: {report_file_path}")
            report_content = _read_file_safe(report_file_path)
            if report_content:
                final_ui_update[markdown_display_comp] = gr.update(value=report_content)
                final_ui_update[markdown_download_comp] = gr.File(value=report_file_path,
                                                                  label=f"Report ({running_task_id}.md)",
                                                                  interactive=True)
            else:
                final_ui_update[markdown_display_comp] = gr.update(
                    value="# Research Complete\n\n*Error reading final report file.*")
        elif final_result_dict and 'report' in final_result_dict:
            logger.info("Using report content directly from agent result.")
            # If agent directly returns report content
            final_ui_update[markdown_display_comp] = gr.update(value=final_result_dict['report'])
            # Cannot offer download if only content is available
            final_ui_update[markdown_download_comp] = gr.update(value=None, label="Download Research Report",
                                                                interactive=False)
        else:
            logger.warning("Final report file not found and not in result dict.")
            final_ui_update[markdown_display_comp] = gr.update(value="# Research Complete\n\n*Final report not found.*")

        yield final_ui_update


    except Exception as e:
        logger.error(f"Error during Deep Research Agent execution: {e}", exc_info=True)
        gr.Error(f"Research failed: {e}")
        yield {markdown_display_comp: gr.update(value=f"# Research Failed\n\n**Error:**\n```\n{e}\n```")}

    finally:
        # --- 8. Final UI Reset ---
        webui_manager.dr_current_task = None  # Clear task reference
        webui_manager.dr_task_id = None  # Clear running task ID

        yield {
            start_button_comp: gr.update(value="▶️ Run", interactive=True),
            stop_button_comp: gr.update(interactive=False),
            research_task_comp: gr.update(interactive=True),
            resume_task_id_comp: gr.update(value="", interactive=True),
            parallel_num_comp: gr.update(interactive=True),
            save_dir_comp: gr.update(interactive=True),
            # Keep download button enabled if file exists
            markdown_download_comp: gr.update() if report_file_path and os.path.exists(report_file_path) else gr.update(
                interactive=False)
        }


async def stop_deep_research(webui_manager: WebuiManager) -> Dict[Component, Any]:
    """Handles the Stop button click."""
    logger.info("Stop button clicked for Deep Research.")
    agent = webui_manager.dr_agent
    task = webui_manager.dr_current_task
    task_id = webui_manager.dr_task_id
    base_save_dir = webui_manager.dr_save_dir

    stop_button_comp = webui_manager.get_component_by_id("deep_research_agent.stop_button")
    start_button_comp = webui_manager.get_component_by_id("deep_research_agent.start_button")
    markdown_display_comp = webui_manager.get_component_by_id("deep_research_agent.markdown_display")
    markdown_download_comp = webui_manager.get_component_by_id("deep_research_agent.markdown_download")

    final_update = {
        stop_button_comp: gr.update(interactive=False, value="⏹️ Stopping...")
    }

    if agent and task and not task.done():
        logger.info("Signalling DeepResearchAgent to stop.")
        try:
            # Assuming stop is synchronous or sets a flag quickly
            await agent.stop()
        except Exception as e:
            logger.error(f"Error calling agent.stop(): {e}")

        # The run_deep_research loop should detect the stop and exit.
        # We yield an intermediate "Stopping..." state. The final reset is done by run_deep_research.

        # Try to show the final report if available after stopping
        await asyncio.sleep(1.5)  # Give agent a moment to write final files potentially
        report_file_path = None
        if task_id and base_save_dir:
            report_file_path = os.path.join(base_save_dir, str(task_id), "report.md")

        if report_file_path and os.path.exists(report_file_path):
            report_content = _read_file_safe(report_file_path)
            if report_content:
                final_update[markdown_display_comp] = gr.update(
                    value=report_content + "\n\n---\n*Research stopped by user.*")
                final_update[markdown_download_comp] = gr.File(value=report_file_path, label=f"Report ({task_id}.md)",
                                                               interactive=True)
            else:
                final_update[markdown_display_comp] = gr.update(
                    value="# Research Stopped\n\n*Error reading final report file after stop.*")
        else:
            final_update[markdown_display_comp] = gr.update(value="# Research Stopped by User")

        # Keep start button disabled, run_deep_research finally block will re-enable it.
        final_update[start_button_comp] = gr.update(interactive=False)

    else:
        logger.warning("Stop clicked but no active research task found.")
        # Reset UI state just in case
        final_update = {
            start_button_comp: gr.update(interactive=True),
            stop_button_comp: gr.update(interactive=False),
            webui_manager.get_component_by_id("deep_research_agent.research_task"): gr.update(interactive=True),
            webui_manager.get_component_by_id("deep_research_agent.resume_task_id"): gr.update(interactive=True),
            webui_manager.get_component_by_id("deep_research_agent.max_iteration"): gr.update(interactive=True),
            webui_manager.get_component_by_id("deep_research_agent.max_query"): gr.update(interactive=True),
        }

    return final_update


async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server.
    """
    if hasattr(webui_manager, "dr_agent") and webui_manager.dr_agent:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.dr_agent.close_mcp_client()

    if not mcp_file or not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)


def create_deep_research_agent_tab(webui_manager: WebuiManager):
    """
    Creates a deep research agent tab
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Row():
            mcp_json_file = gr.File(label="MCP server json", interactive=True, file_types=[".json"])
            mcp_server_config = gr.Textbox(label="MCP server", lines=6, interactive=True, visible=False)

    with gr.Group():
        research_task = gr.Textbox(label="Research Task", lines=5,
                                   value="Give me a detailed travel plan to Switzerland from June 1st to 10th.",
                                   interactive=True)
        with gr.Row():
            resume_task_id = gr.Textbox(label="Resume Task ID", value="",
                                        interactive=True)
            parallel_num = gr.Number(label="Parallel Agent Num", value=1,
                                     precision=0,
                                     interactive=True)
            max_query = gr.Textbox(label="Research Save Dir", value="./tmp/deep_research",
                                   interactive=True)
    with gr.Row():
        stop_button = gr.Button("⏹️ Stop", variant="stop", scale=2)
        start_button = gr.Button("▶️ Run", variant="primary", scale=3)
    with gr.Group():
        markdown_display = gr.Markdown(label="Research Report")
        markdown_download = gr.File(label="Download Research Report", interactive=False)
    tab_components.update(
        dict(
            research_task=research_task,
            parallel_num=parallel_num,
            max_query=max_query,
            start_button=start_button,
            stop_button=stop_button,
            markdown_display=markdown_display,
            markdown_download=markdown_download,
            resume_task_id=resume_task_id,
            mcp_json_file=mcp_json_file,
            mcp_server_config=mcp_server_config,
        )
    )
    webui_manager.add_components("deep_research_agent", tab_components)
    webui_manager.init_deep_research_agent()

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        update_dict = await update_mcp_server(mcp_file, webui_manager)
        yield update_dict

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )

    dr_tab_outputs = list(tab_components.values())
    all_managed_inputs = set(webui_manager.get_components())

    # --- Define Event Handler Wrappers ---
    async def start_wrapper(comps: Dict[Component, Any]) -> AsyncGenerator[Dict[Component, Any], None]:
        async for update in run_deep_research(webui_manager, comps):
            yield update

    async def stop_wrapper() -> AsyncGenerator[Dict[Component, Any], None]:
        update_dict = await stop_deep_research(webui_manager)
        yield update_dict

    # --- Connect Handlers ---
    start_button.click(
        fn=start_wrapper,
        inputs=all_managed_inputs,
        outputs=dr_tab_outputs
    )

    stop_button.click(
        fn=stop_wrapper,
        inputs=None,
        outputs=dr_tab_outputs
    )

```

---

## load_save_config_tab.py

**Path:** `src/webui/components/load_save_config_tab.py`

**Lines:** 50

```python
import gradio as gr
from gradio.components import Component

from src.webui.webui_manager import WebuiManager
from src.utils import config


def create_load_save_config_tab(webui_manager: WebuiManager):
    """
    Creates a load and save config tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    config_file = gr.File(
        label="Load UI Settings from json",
        file_types=[".json"],
        interactive=True
    )
    with gr.Row():
        load_config_button = gr.Button("Load Config", variant="primary")
        save_config_button = gr.Button("Save UI Settings", variant="primary")

    config_status = gr.Textbox(
        label="Status",
        lines=2,
        interactive=False
    )

    tab_components.update(dict(
        load_config_button=load_config_button,
        save_config_button=save_config_button,
        config_status=config_status,
        config_file=config_file,
    ))

    webui_manager.add_components("load_save_config", tab_components)

    save_config_button.click(
        fn=webui_manager.save_config,
        inputs=set(webui_manager.get_components()),
        outputs=[config_status]
    )

    load_config_button.click(
        fn=webui_manager.load_config,
        inputs=[config_file],
        outputs=webui_manager.get_components(),
    )


```

---

## webui.py

**Path:** `webui.py`

**Lines:** 19

```python
from dotenv import load_dotenv
load_dotenv()
import argparse
from src.webui.interface import theme_map, create_ui


def main():
    parser = argparse.ArgumentParser(description="Gradio WebUI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Windows 98 RGB", choices=theme_map.keys(), help="Theme to use for the UI")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.queue().launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()

```

---

