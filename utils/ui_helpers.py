from functools import lru_cache
from pathlib import Path
import streamlit as st

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

BASE_DIR = Path(__file__).resolve().parent.parent


def adjust_hex_color(color: str, factor: float) -> str:
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