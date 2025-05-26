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
/* --- RGB ANIMATIONS (largely reusable) --- */
@keyframes rgbTextGlow { /* Brighter, more focused glow for dark theme */
  0%, 100% { text-shadow: 0 0 1px #fff, 0 0 3px #ff0000, 0 0 5px #ff0000; filter: hue-rotate(0deg); }
  16% { text-shadow: 0 0 1px #fff, 0 0 3px #ff7700, 0 0 5px #ff7700; filter: hue-rotate(30deg); }
  32% { text-shadow: 0 0 1px #fff, 0 0 3px #ffff00, 0 0 5px #ffff00; filter: hue-rotate(60deg); }
  48% { text-shadow: 0 0 1px #fff, 0 0 3px #00ff00, 0 0 5px #00ff00; filter: hue-rotate(120deg); }
  64% { text-shadow: 0 0 1px #fff, 0 0 3px #00ffff, 0 0 5px #00ffff; filter: hue-rotate(180deg); }
  80% { text-shadow: 0 0 1px #fff, 0 0 3px #0000ff, 0 0 5px #0000ff; filter: hue-rotate(240deg); }
  96% { text-shadow: 0 0 1px #fff, 0 0 3px #ff00ff, 0 0 5px #ff00ff; filter: hue-rotate(300deg); }
}

@keyframes rgbBorderGlow { /* Intense border glow */
  0%, 100% { border-color: #ff0000 !important; box-shadow: 0 0 2px #ff0000, 0 0 4px #ff0000, inset 0 0 1px #ff000088; }
  16% { border-color: #ff7700 !important; box-shadow: 0 0 2px #ff7700, 0 0 4px #ff7700, inset 0 0 1px #ff770088; }
  32% { border-color: #ffff00 !important; box-shadow: 0 0 2px #ffff00, 0 0 4px #ffff00, inset 0 0 1px #ffff0088; }
  48% { border-color: #00ff00 !important; box-shadow: 0 0 2px #00ff00, 0 0 4px #00ff00, inset 0 0 1px #00ff0088; }
  64% { border-color: #00ffff !important; box-shadow: 0 0 2px #00ffff, 0 0 4px #00ffff, inset 0 0 1px #00ffff88; }
  80% { border-color: #0000ff !important; box-shadow: 0 0 2px #0000ff, 0 0 4px #0000ff, inset 0 0 1px #0000ff88; }
  96% { border-color: #ff00ff !important; box-shadow: 0 0 2px #ff00ff, 0 0 4px #ff00ff, inset 0 0 1px #ff00ff88; }
}

@keyframes rgbBackgroundGradient { /* For accents */
    0%, 100% { background-position: 0% 50%; filter: hue-rotate(0deg) brightness(1.2) saturate(2); }
    50% { background-position: 100% 50%; filter: hue-rotate(180deg) brightness(1.2) saturate(2); }
}

@keyframes rgbShadowPulse { /* More subtle shadow for dark theme */
  0%, 100% { box-shadow: 1px 1px 0 #ff000033, 2px 2px 0 #000000cc, 0 0 5px #ff000088; filter: hue-rotate(0deg); }
  25% { box-shadow: 1px 1px 0 #00ff0033, 2px 2px 0 #000000cc, 0 0 5px #00ff0088; filter: hue-rotate(90deg); }
  50% { box-shadow: 1px 1px 0 #0000ff33, 2px 2px 0 #000000cc, 0 0 5px #0000ff88; filter: hue-rotate(180deg); }
  75% { box-shadow: 1px 1px 0 #ffff0033, 2px 2px 0 #000000cc, 0 0 5px #ffff0088; filter: hue-rotate(270deg); }
}

/* --- GLOBAL & BODY --- */
body, html {
    margin: 0; padding: 0;
    font-family: "Pixelated MS Sans Serif", "MS Sans Serif", "Perfect DOS VGA", Tahoma, monospace !important;
    background: #080a10 linear-gradient(135deg, #000000, #080a10, #0d0f18, #080a10, #000000); /* Dark, subtly animated space */
    background-size: 300% 300%;
    animation: rgbBackgroundGradient 30s ease-in-out infinite alternate; /* Slower, more atmospheric */
    color: #c0d0ff; /* Default light blue/lavender text */
    overflow: auto;
}

body *, /* Apply font more broadly */
.gradio-container, input, textarea, select, button, .gr-button, .gr-input-text input, .gr-dropdown select,
.gr-label span, .gr-checkbox label span, .gr-radio label span, .gr-tabs button,
.label-span, .prose {
    font-family: "Pixelated MS Sans Serif", "MS Sans Serif", "Perfect DOS VGA", Tahoma, monospace !important;
    font-size: 13px; /* Slightly larger for dark theme readability */
    -webkit-font-smoothing: none; font-smooth: never;
}

/* --- MAIN CONTAINER (The "Desktop") --- */
.gradio-container {
    background-color: #10141c !important; /* Dark blue/grey base */
    padding: 3px;
    border: 2px outset #202838; /* Lighter shade for top/left bevel */
    border-right-color: #000000; /* Darker for bottom/right bevel */
    border-bottom-color: #000000;
    box-shadow: 0 0 0 2px black, 0 0 0 4px transparent;
    animation: rgbBorderGlow 6s linear infinite; /* Container border RGB */
    border-radius: 0px;
    margin: 10px auto; max-width: 1200px;
}

/* --- APP TITLE BAR --- */
.title-bar {
    background: linear-gradient(90deg, #ff0000, #dd6600, #bbbb00, #00aa00, #00aaaa, #0000aa, #aa00aa); /* More saturated */
    background-size: 200% 100%;
    color: white; text-align: center; font-weight: bold; padding: 4px 0; font-size: 16px !important;
    border: 1px solid #000; margin: 3px;
    animation: rgbBackgroundGradient 4s linear infinite, rgbTextGlow 3s ease-in-out infinite alternate;
    text-shadow: 1px 1px 0px #000A;
}

/* --- WINDOW-LIKE ELEMENTS (GROUPS, TAB CONTENT) --- */
.window-body, .gr-group, .gr-panel, .gr-tabitem > .p-4.gap-4 {
    background: #181c28 !important; /* Slightly lighter dark blue/grey for windows */
    border: 2px outset #2a3040;
    border-right-color: #080a10;
    border-bottom-color: #080a10;
    box-shadow: 1px 1px 0px #00000088;
    padding: 10px; margin-top:0; margin-bottom: 8px; color: #c0d0ff; border-radius: 0px;
}
/* Group Title Bar - Classic Blue on Dark */
.gr-group > .gr-label span, .gr-block.gr-label > .label-span {
    display: block;
    background: linear-gradient(to right, #000050, #0000A0, #1040C0, #0000A0, #000050); /* Richer blue gradient */
    color: white !important; padding: 3px 6px; margin: -10px -10px 8px -10px;
    border-bottom: 1px solid #000030; font-weight: bold; font-size: 12px !important;
    text-shadow: 1px 1px #000000AA;
    position: relative; overflow: hidden;
    animation: rgbTextGlow 5s infinite alternate; /* Title text RGB glow */
}
.gr-group > .gr-label span::after, .gr-block.gr-label > .label-span::after {
    content: ''; position: absolute; top:0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(120,150,255,0.2), transparent);
    animation: shimmer 2.5s infinite; /* Faster shimmer */
}
@keyframes shimmer { 0% {left: -100%;} 50% {left: 100%;} 100% {left: 100%;} }


/* --- TABS --- */
.gr-tabs > div[role="tablist"] { background: #10141c; border-bottom: none; padding: 2px 2px 0 2px; }
.gr-tabs button {
    background: #181c28 !important; color: #a0b0dd !important; /* Darker tab, lighter text */
    border: 2px outset #2a3040 !important;
    border-right-color: #080a10 !important; border-bottom-color: #080a10 !important;
    border-bottom-style: solid !important;
    margin-right: 1px; padding: 4px 10px !important; box-shadow: none;
    font-size: 12px !important; border-radius: 0 !important; position: relative; min-height: 24px;
}
.gr-tabs button.selected {
    background: #202838 !important; /* Selected tab slightly lighter */
    color: #e0e8ff !important;
    border-bottom-style: none !important; z-index: 1;
    border-top-style: solid !important; border-left-style: solid !important; border-right-style: solid !important;
    border-width: 2px !important;
    animation: rgbBorderGlow 2.5s linear infinite;
    padding-top: 4px !important; padding-bottom: 6px !important;
}
.gr-tabs button:not(.selected):hover {
    background: #202838 !important; color: #c0d0ff !important;
    animation: rgbBorderGlow 3s linear infinite alternate;
}
.gr-tabitem > .p-4.gap-4 { border-top: none; margin-top: -2px; padding-top: 12px; }

/* --- INPUTS, TEXTAREAS, SELECTS --- */
input[type="text"], input[type="password"], input[type="number"], textarea,
.gr-input-text input, .gr-textarea textarea, .gr-number input {
    background: #000000 !important; color: #00ff88 !important; /* Black bg, green text */
    border: 2px inset #202020 !important;
    border-top-color: #000000 !important; border-left-color: #000000 !important;
    border-right-color: #303030 !important; border-bottom-color: #303030 !important;
    padding: 3px 5px !important; border-radius: 0px !important; box-shadow: none;
    animation: rgbTextGlow 7s ease-in-out infinite; /* Subtle text animation for inputs */
}
*:focus { outline: none !important; }

input:focus, textarea:focus, .gr-dropdown select:focus {
    border-style: solid !important;
    animation: rgbBorderGlow 1.5s linear infinite, rgbTextGlow 7s ease-in-out infinite !important;
}

/* Dropdowns */
.gr-dropdown select {
    appearance: none; -webkit-appearance: none; -moz-appearance: none;
    background-color: #000000 !important; color: #00ff88 !important;
    padding-right: 20px !important;
}
.gr-dropdown::after { /* Custom arrow button area */
    content: ""; position: absolute;
    top: 1px; right: 1px; bottom: 1px; width: 17px;
    background: #181c28;
    border: 2px outset #2a3040; border-right-color: #080a10; border-bottom-color: #080a10;
    pointer-events: none; display: flex; align-items: center; justify-content: center;
    animation: rgbBackgroundGradient 3s linear infinite alternate;
}
.gr-dropdown::before { /* The actual arrow */
    content: "▼"; font-size: 9px; color: #000;
    position: absolute; top: 50%; right: 5px; transform: translateY(-50%);
    pointer-events: none; z-index: 1;
    text-shadow: 0 0 2px transparent; animation: rgbTextGlow 4s linear infinite;
}
.gr-dropdown .options {
    background-color: #000000 !important; border: 1px solid #333 !important; color: #00ff88 !important; border-radius: 0px !important;
    animation: rgbBorderGlow 5s linear infinite; /* Border of dropdown list */
}
.gr-dropdown .options .option:hover, .gr-dropdown .options .option.selected {
    background: linear-gradient(90deg, #550000, #005500, #000055) !important; /* Darker RGB select */
    color: #FFFFFF !important; text-shadow: 1px 1px #000;
    animation: rgbBackgroundGradient 2s linear infinite, rgbTextGlow 3s linear infinite;
}

/* --- BUTTONS --- */
button, .gr-button {
    background: #202838 !important; color: #c0d0ff !important;
    border: 2px outset #303a48 !important;
    border-right-color: #10141c !important; border-bottom-color: #10141c !important;
    box-shadow: 1px 1px 0px #00000088;
    padding: 3px 12px !important; border-radius: 0px !important;
    font-size: 12px !important; min-height: 25px; text-shadow: 1px 1px #0005;
}
button:hover, .gr-button:hover, button:focus, .gr-button:focus {
    background: #283040 !important; color: #e0e8ff !important;
    border-style: solid !important;
    animation: rgbBorderGlow 2s linear infinite alternate, rgbTextGlow 4s linear infinite alternate;
}
button:active, .gr-button:active {
    border-style: inset !important; background: #181c28 !important; color: #a0b0dd !important;
    padding: 4px 11px 2px 13px !important; animation: none; text-shadow: none;
}
.gr-button.gr-button-primary { /* Primary button even more RGB */
    animation: rgbBorderGlow 1.5s linear infinite, rgbShadowPulse 2.5s linear infinite, rgbTextGlow 3s linear infinite alternate;
    border-style: solid !important;
}
.gr-button.gr-button-primary:active { animation: none; box-shadow: inset 1px 1px 2px #000; }

/* --- CHECKBOXES & RADIO --- */
.gr-checkbox input[type="checkbox"], .gr-radio input[type="radio"] {
    appearance: none; -webkit-appearance: none;
    width: 13px; height: 13px; background-color: #000;
    border: 1px solid #444; box-shadow: inset 1px 1px 0px #222, inset -1px -1px 0px #000;
    position: relative; vertical-align: -2px; margin-right: 5px; cursor: pointer; border-radius: 0px;
}
.gr-checkbox input[type="checkbox"]:checked::before {
    content: ''; display: block; width: 7px; height: 4px;
    border-left-width: 2px; border-bottom-width: 2px; border-style: solid;
    transform: translate(1.5px, 2.5px) rotate(-45deg);
    animation: rgbBorderGlow 2s linear infinite;
}
.gr-radio input[type="radio"] { border-radius: 50%; }
.gr-radio input[type="radio"]:checked::before {
    content: ''; display: block; width: 7px; height: 7px; border-radius: 50%;
    position: absolute; top: 2px; left: 2px;
    animation: rgbBackgroundGradient 1.5s linear infinite;
}
.gr-checkbox label span, .gr-radio label span { color: #c0d0ff !important; font-size: 12px !important; }

/* --- SLIDERS --- */
.gr-slider input[type="range"] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 19px;
    background: transparent; outline: none; padding:0; margin: 4px 0; position: relative;
}
.gr-slider input[type="range"]::before { /* Track */
    content: ''; position: absolute; left: 0; top: 8px; width: 100%; height: 3px;
    background: #000; border: 1px solid; border-color: #000 #333 #333 #000;
}
.gr-slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none; width: 10px; height: 21px;
    background: #202838; border: 1px outset #303a48;
    border-right-color: #10141c; border-bottom-color: #10141c;
    cursor: pointer; position: relative; top: -1px; box-shadow: 1px 1px 0 #0008;
    animation: rgbBackgroundGradient 1.5s ease-in-out infinite, rgbBorderGlow 3s linear infinite alternate;
}
/* Similar for ::-moz-range-thumb if needed */
.gr-slider .value_input input { width: 40px !important; font-size: 11px !important; padding: 2px !important; }


/* --- CHATBOT --- */
.gr-chatbot {
    background-color: #0a0c14 !important; /* Very dark chatbot background */
    border: 2px inset #000000 !important; border-top-color: #000 !important; border-left-color: #000 !important;
    border-right-color: #181c28 !important; border-bottom-color: #181c28 !important;
    padding: 5px; border-radius: 0px;
}
.gr-chatbot .message {
    padding: 6px 10px !important; border-radius: 0px !important;
    font-size: 12px !important; line-height: 1.4;
    box-shadow: 1px 1px 0 #0005; margin-bottom: 5px !important;
    border-style: solid !important; border-width: 1px !important;
    animation: rgbBorderGlow 4s linear infinite alternate;
}
.gr-chatbot .message.user { background-color: #101828 !important; color: #c0e0ff !important; }
.gr-chatbot .message.bot { background-color: #182030 !important; color: #d0f0d0 !important; }
.gr-chatbot pre {
    background-color: #000 !important; padding: 5px !important;
    color: lime !important; font-family: "Perfect DOS VGA", "Fixedsys", monospace !important; font-size: 11px !important;
    border-radius: 0px; border-style: solid !important; border-width: 1px !important;
    animation: rgbBorderGlow 3s ease-in-out infinite, rgbTextGlow 5s ease-in-out infinite alternate;
    white-space: pre-wrap; word-break: break-all;
}
.gr-chatbot code { color: lime !important; background: transparent !important; }
.gr-chatbot .avatar-container { display: none !important; }
.gr-chatbot .copy-button { filter: none !important; opacity: 0.8; animation: rgbTextGlow 3s linear infinite; }


/* --- LABELS for components --- */
.gr-form > .gr-label > .label-span, .gr-block.gr-label > .label-span, label.svelte-1gfkn6j,
div[class*="Label"] span[class*="label"], .gr-block > label > span {
    color: #a0b0dd !important; font-weight: normal !important;
    font-size: 12px !important; text-shadow: 1px 1px #000A; padding-bottom: 2px;
}

/* --- SCROLLBARS - Dark Retro RGB --- */
::-webkit-scrollbar { width: 17px; height: 17px; }
::-webkit-scrollbar-track { background: #080a10; }
::-webkit-scrollbar-thumb {
    background: #181c28; border: 2px outset #2a3040;
    border-right-color: #080a10; border-bottom-color: #080a10;
    animation: rgbBackgroundGradient 2.5s linear infinite;
}
::-webkit-scrollbar-thumb:hover { filter: brightness(1.2); }
::-webkit-scrollbar-button {
    background: #181c28; border: 2px outset #2a3040;
    border-right-color: #080a10; border-bottom-color: #080a10;
    display: block; height: 17px; width: 17px;
    animation: rgbBackgroundGradient 4s linear infinite alternate;
}
::-webkit-scrollbar-button:hover { filter: brightness(1.15); }
::-webkit-scrollbar-button:active { border-style: inset; animation: none; }
/* Arrows (darker background, brighter arrow) */
::-webkit-scrollbar-button:vertical:decrement { background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns="http://www.w3.org/2000/svg" width="9" height="5"%3E%3Cpath d="M4.5 0L9 5H0z" fill="%238090c0"/%3E%3C/svg%3E'); background-repeat: no-repeat; background-position: center; }
::-webkit-scrollbar-button:vertical:increment { background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns="http://www.w3.org/2000/svg" width="9" height="5"%3E%3Cpath d="M4.5 5L9 0H0z" fill="%238090c0"/%3E%3C/svg%3E'); background-repeat: no-repeat; background-position: center; }
::-webkit-scrollbar-button:horizontal:decrement { background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns="http://www.w3.org/2000/svg" width="5" height="9"%3E%3Cpath d="M0 4.5L5 9V0z" fill="%238090c0"/%3E%3C/svg%3E'); background-repeat: no-repeat; background-position: center; }
::-webkit-scrollbar-button:horizontal:increment { background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns="http://www.w3.org/2000/svg" width="5" height="9"%3E%3Cpath d="M5 4.5L0 9V0z" fill="%238090c0"/%3E%3C/svg%3E'); background-repeat: no-repeat; background-position: center; }
::-webkit-scrollbar-corner { background: #080a10; animation: rgbBackgroundGradient 5s linear infinite; }


/* --- MISC & MARKDOWN --- */
.prose { color: #c0d0ff !important; }
.prose a { color: #80a0ff !important; text-decoration: underline !important; animation: rgbTextGlow 3s linear infinite; }
.prose strong { color: #e0f0ff !important; font-weight: bold; animation: rgbTextGlow 5s linear infinite alternate; }
.prose code {
    background-color: #000000 !important; color: #88ffcc !important; padding: 0.1em 0.4em;
    border-style: solid !important; border-width: 1px !important;
    animation: rgbBorderGlow 4s linear infinite, rgbTextGlow 6s linear infinite;
}
.prose pre {
    border-style: solid !important; border-width: 1px !important; animation: rgbBorderGlow 3s linear infinite;
    background-color: #000 !important; /* Terminal style for pre */
}
.prose pre code { /* Terminal style text in pre */
    background-color: transparent !important; border: none; animation: none; color: lime !important;
    font-family: "Perfect DOS VGA", "Fixedsys", monospace !important;
}
.prose h1, .prose h2, .prose h3, .prose h4 {
    color: #e0e8ff !important; border-bottom: 1px solid #303a48;
    animation: rgbTextGlow 6s linear infinite;
}
.prose ul > li::before { background-color: #80a0ff !important; animation: rgbBackgroundGradient 2s linear infinite; }
.prose blockquote {
    border-left-style: solid !important; border-left-width: 4px !important;
    animation: rgbBorderGlow 3s linear infinite; color: #a0b0dd !important;
    background-color: #10141cAA !important;
}

/* Subtle Scanlines for CRT effect */
body::after {
    content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: repeating-linear-gradient(transparent, transparent 1px, rgba(0,0,0,0.25) 2px, rgba(0,0,0,0.25) 3px);
    opacity: 0.4; pointer-events: none; z-index: 9999;
    mix-blend-mode: multiply; /* Darkens underlying colors for scanlines */
}


</style>
'''

    ui_manager = WebuiManager()

    with gr.Blocks(title="Browser Use WebUI - Dark Retro RGB Overdrive", theme=theme_map[theme_name], head=head_html) as demo:
        with gr.Row():
            gr.Markdown("# 🖥️ Browser Use WebUI", elem_classes=["title-bar"])

        with gr.Tabs(elem_id="main_tabs") as tabs:
            with gr.TabItem("⚙️ Agent Settings"):
                with gr.Group(elem_classes=["window-body"]):
                    create_agent_settings_tab(ui_manager)

            with gr.TabItem("🌐 Browser Settings"):
                with gr.Group(elem_classes=["window-body"]):
                    create_browser_settings_tab(ui_manager)

            with gr.TabItem("🤖 Run Agent"):
                with gr.Group(elem_classes=["window-body"]):
                    create_browser_use_agent_tab(ui_manager)

            with gr.TabItem("🎁 Agent Marketplace"):
                with gr.Group(elem_classes=["window-body"]):
                    gr.Markdown("### Agent Marketplace")
                    with gr.Tabs():
                        with gr.TabItem("Deep Research"):
                             with gr.Group(elem_classes=["window-body"]):
                                create_deep_research_agent_tab(ui_manager)

            with gr.TabItem("📁 Load & Save Config"):
                with gr.Group(elem_classes=["window-body"]):
                    create_load_save_config_tab(ui_manager)
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(debug=True)
