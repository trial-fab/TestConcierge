"""UI helpers for theme colors, CSS injection, and auto-scrolling."""
from functools import lru_cache
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

BASE_DIR = Path(__file__).resolve().parent.parent


def adjust_hex_color(color: str, factor: float) -> str:
    """Lighten (factor>0) or darken (factor<0) a hex color."""
    if not color:
        return color
    hex_value = color.strip().lstrip("#")
    if len(hex_value) not in (3, 6):
        return color
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)
    comps = [int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]
    out = []
    for comp in comps:
        if factor >= 0:
            comp = comp + (255 - comp) * min(factor, 1)
        else:
            comp = comp * max(1 + factor, 0)
        out.append(int(max(0, min(255, round(comp)))))
    return "#{:02x}{:02x}{:02x}".format(*out)


@lru_cache(maxsize=1)
def load_theme_colors() -> dict[str, str]:
    """Load theme colors from .streamlit/config.toml."""
    defaults = {
        "primary": "#006747",
        "background": "#f7f8f7",
        "secondary": "#e7efe5",
        "text": "#111111",
    }
    config_path = BASE_DIR / ".streamlit" / "config.toml"
    if config_path.exists():
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            theme = data.get("theme", {})
            defaults["primary"] = theme.get("primaryColor", defaults["primary"])
            defaults["background"] = theme.get("backgroundColor", defaults["background"])
            defaults["secondary"] = theme.get("secondaryBackgroundColor", defaults["secondary"])
            defaults["text"] = theme.get("textColor", defaults["text"])
        except (tomllib.TOMLDecodeError, OSError):
            pass
    defaults["primary_dark"] = adjust_hex_color(defaults["primary"], -0.2)
    return defaults


def inject_global_styles() -> None:
    """Inject global CSS styles with theme variables."""
    css_path = BASE_DIR / "styles.css"
    chunks = []
    colors = load_theme_colors()
    chunks.append(
        f"""
:root {{
  --usf-green: {colors['primary']};
  --usf-dark-green: {colors['primary_dark']};
  --usf-light-bg: {colors['background']};
  --usf-secondary-bg: {colors['secondary']};
  --usf-text-color: {colors['text']};
}}
"""
    )
    if css_path.exists():
        chunks.append(css_path.read_text(encoding="utf-8"))
    st.markdown("<style>" + "\n".join(chunks) + "</style>", unsafe_allow_html=True)


def scroll_chat_to_bottom() -> None:
    """
    Smart auto-scroll that sticks to bottom when new messages appear,
    but stops when user scrolls up manually (like ChatGPT).

    Behavior:
    - Always scrolls to bottom when new content appears
    - Detects user scroll-up and disables auto-scroll
    - Re-enables auto-scroll when user manually scrolls to bottom
    """
    # Create a unique token based on current state to trigger scroll on changes
    messages_len = len(st.session_state.get("messages", []))
    show_email = int(bool(st.session_state.get("show_email_builder")))
    show_meeting = int(bool(st.session_state.get("show_meeting_builder")))
    show_picker = int(bool(st.session_state.get("show_tool_picker")))

    token = f"{messages_len}-{show_email}-{show_meeting}-{show_picker}"

    components.html(
        f"""
        <div style="display:none" data-scroll-token="{token}"></div>
        <script>
        (function() {{
            // Storage key for scroll lock state
            const SCROLL_LOCK_KEY = 'chatScrollLocked';

            // Check if user has manually scrolled up
            let isScrollLocked = sessionStorage.getItem(SCROLL_LOCK_KEY) === 'true';

            const scrollToBottom = () => {{
                try {{
                    const doc = window.parent?.document || document;
                    const mainBlock = doc.querySelector('.main .block-container');
                    const scrollTarget = mainBlock || doc.documentElement || doc.body;

                    if (scrollTarget) {{
                        // Check if we're near the bottom (within 100px)
                        const isNearBottom = scrollTarget.scrollHeight - scrollTarget.scrollTop - scrollTarget.clientHeight < 100;

                        // Only auto-scroll if not locked OR if user is near bottom
                        if (!isScrollLocked || isNearBottom) {{
                            scrollTarget.scrollTo({{
                                top: scrollTarget.scrollHeight,
                                behavior: 'smooth'
                            }});

                            // If we successfully scrolled to bottom, unlock
                            if (isNearBottom) {{
                                isScrollLocked = false;
                                sessionStorage.setItem(SCROLL_LOCK_KEY, 'false');
                            }}
                        }}
                    }}
                }} catch (err) {{
                    // Silently handle scroll errors
                }}
            }};

            // Set up scroll listener to detect user scrolling up
            const setupScrollListener = () => {{
                try {{
                    const doc = window.parent?.document || document;
                    const mainBlock = doc.querySelector('.main .block-container');
                    const scrollTarget = mainBlock || doc.documentElement || doc.body;

                    if (scrollTarget && !scrollTarget.dataset.scrollListenerAttached) {{
                        scrollTarget.dataset.scrollListenerAttached = 'true';

                        let scrollTimeout;
                        scrollTarget.addEventListener('scroll', () => {{
                            clearTimeout(scrollTimeout);
                            scrollTimeout = setTimeout(() => {{
                                const atBottom = scrollTarget.scrollHeight - scrollTarget.scrollTop - scrollTarget.clientHeight < 50;

                                if (atBottom) {{
                                    // User scrolled to bottom manually - unlock auto-scroll
                                    isScrollLocked = false;
                                    sessionStorage.setItem(SCROLL_LOCK_KEY, 'false');
                                }} else {{
                                    // User scrolled up - lock auto-scroll
                                    const currentScroll = scrollTarget.scrollTop;
                                    const maxScroll = scrollTarget.scrollHeight - scrollTarget.clientHeight;

                                    // Only lock if user scrolled up (not at bottom)
                                    if (currentScroll < maxScroll - 50) {{
                                        isScrollLocked = true;
                                        sessionStorage.setItem(SCROLL_LOCK_KEY, 'true');
                                    }}
                                }}
                            }}, 150);
                        }}, {{ passive: true }});
                    }}
                }} catch (err) {{
                    // Silently handle listener setup errors
                }}
            }};

            // Execute scroll and setup listener
            if (document.readyState === 'complete') {{
                setupScrollListener();
                setTimeout(scrollToBottom, 50);
            }} else {{
                window.addEventListener('load', () => {{
                    setupScrollListener();
                    setTimeout(scrollToBottom, 50);
                }}, {{ once: true }});
            }}
        }})();
        </script>
        """,
        height=0,
    )
