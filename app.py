"""
AURA: The BiE Brand Architect
Virtual Chief Marketing Officer for Beauty by BiE — a luxury skincare brand.

Run with:  streamlit run app.py
"""

import os
import re
import time
import textwrap
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------------------------------
# ENV / API KEY
# ---------------------------------------------------------------------------
load_dotenv()

def _get_api_key() -> str:
    """Resolve the Anthropic API key from Streamlit secrets or env."""
    key = None
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        key = os.getenv("ANTHROPIC_API_KEY")
    return key or ""

# ---------------------------------------------------------------------------
# LLM CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_ID = "anthropic/claude-haiku-4-5-20251001"

def _build_llm(api_key: str) -> LLM:
    return LLM(
        model=MODEL_ID,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.7,
    )

# ---------------------------------------------------------------------------
# CREWAI TOOL — DuckDuckGo web search
# ---------------------------------------------------------------------------
@tool("WebSearch")
def web_search(search_query: str) -> str:
    """Search the internet for current market data, competitor info, and trends."""
    try:
        return DuckDuckGoSearchRun().run(search_query)
    except Exception as exc:
        return f"Search unavailable: {exc}"

# ---------------------------------------------------------------------------
# AGENT DEFINITIONS
# ---------------------------------------------------------------------------

def _create_researcher(llm: LLM) -> Agent:
    return Agent(
        role="Market Intelligence Researcher",
        goal=(
            "Scan the Indian luxury skincare landscape — especially Forest Essentials, "
            "Kama Ayurveda, and emerging D2C brands — to find unmet consumer needs, "
            "white-space opportunities, and 'Blue Ocean' positioning angles for "
            "Beauty by BiE."
        ),
        backstory=(
            "You are a razor-sharp market analyst who spent a decade at Bain & Company "
            "before joining a luxury beauty conglomerate. You think in frameworks "
            "(Porter's Five Forces, Jobs-To-Be-Done) but write with the clarity of a "
            "Financial Times columnist. Your intel has helped launch three unicorn "
            "beauty brands in Asia."
        ),
        llm=llm,
        tools=[web_search],
        verbose=False,
        allow_delegation=False,
        max_iter=15,
    )


def _create_strategist(llm: LLM) -> Agent:
    return Agent(
        role="Brand Strategy Architect",
        goal=(
            "Synthesise the market research into a compelling Brand Launch Bible for "
            "Beauty by BiE. The tone must be luxury-editorial — imagine a Vogue editor "
            "who moonlights as a McKinsey partner. Centre everything around the theme "
            "'Modern Metamorphosis' — transformation through science-backed Ayurveda."
        ),
        backstory=(
            "You are a legendary brand strategist who built the positioning for three "
            "of India's most coveted luxury labels. You trained under Francois Nars, "
            "studied semiotics at Central Saint Martins, and believe that a brand is a "
            "promise wrapped in desire. You write manifestos, not memos."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=15,
    )


def _create_executor(llm: LLM) -> Agent:
    return Agent(
        role="Campaign Execution Director",
        goal=(
            "Translate the Brand Launch Bible into a battle-ready 30-day content "
            "calendar for Beauty by BiE. Cover Instagram Reels, Meta/Facebook Ads, "
            "influencer seedings, and PR moments. Every entry must include the hook, "
            "format, CTA, and target audience segment."
        ),
        backstory=(
            "You ran social-first launches for Glossier, Drunk Elephant, and "
            "Forest Essentials before going independent. You understand the algorithm "
            "intimately — thumb-stop rates, save-to-share ratios, and the 3-second "
            "hook rule. Your calendars don't just plan content; they engineer virality."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=15,
    )

# ---------------------------------------------------------------------------
# TASK DEFINITIONS
# ---------------------------------------------------------------------------

def _create_tasks(
    researcher: Agent,
    strategist: Agent,
    executor: Agent,
    product_concept: str,
) -> list[Task]:

    research_task = Task(
        description=textwrap.dedent(f"""\
            Conduct deep market intelligence for the product concept: **{product_concept}**
            by Beauty by BiE — a premium Indian skincare brand.

            Your deliverables:
            1. Competitive landscape snapshot — top 5 Indian luxury skincare competitors
               (must include Forest Essentials & Kama Ayurveda), their hero products,
               price positioning, and brand narrative gaps.
            2. Consumer insight pulse — what Indian luxury skincare consumers (25-40,
               SEC A/A+) are actively seeking but not finding.
            3. Three 'Blue Ocean' positioning angles that would give Beauty by BiE
               an uncontested market space for this product.
            4. Key ingredient and formulation trends gaining traction globally
               (K-beauty, J-beauty, Ayurveda 2.0) that are relevant to this concept.
        """),
        expected_output=(
            "A structured market intelligence brief (800-1200 words) with numbered "
            "sections, specific competitor references, data points where available, "
            "and three clearly articulated Blue Ocean positioning angles."
        ),
        agent=researcher,
    )

    strategy_task = Task(
        description=textwrap.dedent(f"""\
            Using the market intelligence provided, create the Brand Launch Bible
            for **{product_concept}** by Beauty by BiE.

            The Bible must include:
            1. **Brand Positioning Statement** — one paragraph that captures
               the essence through the 'Modern Metamorphosis' lens.
            2. **Product Story Arc** — the narrative journey from problem to
               transformation. Write it like a Vogue feature.
            3. **Target Persona** — a vivid, named persona (e.g., "Priya, The
               Conscious Luxurist") with psychographics, not just demographics.
            4. **Visual & Verbal Identity Direction** — colour palette mood,
               typography feel, voice guidelines (3 adjectives + 3 'never' words).
            5. **Pricing & Channel Strategy** — recommended MRP range,
               launch channels (D2C site, Nykaa Luxe, select retail), and why.
            6. **Launch Moment Concept** — a single, ownable launch event or
               activation idea that would generate earned media.

            Tone: Luxury editorial meets strategic rigour. Every sentence should
            feel like it belongs in both Vogue Business and a McKinsey deck.
        """),
        expected_output=(
            "A comprehensive Brand Launch Bible (1500-2000 words) with all six "
            "sections clearly delineated, written in the specified luxury-editorial "
            "tone with the 'Modern Metamorphosis' theme threaded throughout."
        ),
        agent=strategist,
        context=[research_task],
    )

    execution_task = Task(
        description=textwrap.dedent(f"""\
            Using the Brand Launch Bible, create a 30-day content calendar
            for the launch of **{product_concept}** by Beauty by BiE.

            Calendar requirements:
            - Organised by Week (Week 1-4), then by Day.
            - Each entry must specify:
              • Platform (Instagram Reels / Meta Ad / Story / Carousel / PR)
              • Content Hook (the first 3 seconds or headline)
              • Format & Duration
              • CTA (Call to Action)
              • Target Segment
            - Include at least:
              • 8 Instagram Reels concepts
              • 4 Meta/Facebook ad creatives (with headline + body copy direction)
              • 2 influencer seeding moments
              • 1 PR / earned-media activation
              • 2 UGC prompts / hashtag challenges
            - Week 1 = Teaser / mystery phase
            - Week 2 = Reveal / education phase
            - Week 3 = Social proof / influencer wave
            - Week 4 = Conversion / urgency phase
        """),
        expected_output=(
            "A detailed 30-day content calendar formatted week-by-week with "
            "every entry containing Platform, Hook, Format, CTA, and Target "
            "Segment. Minimum 20 distinct content entries across the 30 days."
        ),
        agent=executor,
        context=[research_task, strategy_task],
    )

    return [research_task, strategy_task, execution_task]

# ---------------------------------------------------------------------------
# CREW ASSEMBLY & EXECUTION
# ---------------------------------------------------------------------------

def run_aura_crew(product_concept: str, api_key: str, progress_callback=None):
    """Build the three-agent crew and kick off the sequential workflow."""
    llm = _build_llm(api_key)

    researcher = _create_researcher(llm)
    strategist = _create_strategist(llm)
    executor = _create_executor(llm)

    tasks = _create_tasks(researcher, strategist, executor, product_concept)

    crew = Crew(
        agents=[researcher, strategist, executor],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )

    if progress_callback:
        progress_callback(10, "Assembling the AURA agent team ...")

    result = crew.kickoff()

    if progress_callback:
        progress_callback(95, "Finalising strategy document ...")

    return result

# ---------------------------------------------------------------------------
# PDF GENERATION
# ---------------------------------------------------------------------------

class AuraPDF(FPDF):
    """Custom PDF with luxury dark header/footer."""

    GOLD = (191, 155, 81)
    DARK = (18, 18, 18)
    LIGHT = (240, 236, 228)
    WHITE = (255, 255, 255)

    def header(self):
        self.set_fill_color(*self.DARK)
        self.rect(0, 0, 210, 28, "F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*self.GOLD)
        self.set_y(8)
        self.cell(0, 10, "AURA  |  Beauty by BiE", align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-18)
        self.set_fill_color(*self.DARK)
        self.rect(0, self.get_y() - 2, 210, 20, "F")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.GOLD)
        self.cell(0, 10, f"Confidential Strategy Document  |  Page {self.page_no()}", align="C")


def _sanitize_text(text: str) -> str:
    """Replace characters that the default Helvetica encoding cannot handle."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "*",    # bullet
        "\u00a0": " ",    # non-breaking space
        "\u2010": "-",    # hyphen
        "\u2011": "-",    # non-breaking hyphen
        "\u2012": "-",    # figure dash
        "\u00b7": "*",    # middle dot
        "\u25cf": "*",    # black circle
        "\u25cb": "o",    # white circle
        "\u2023": ">",    # triangular bullet
        "\u00e9": "e",    # accented e
        "\u00e8": "e",    # accented e (grave)
        "\u00e0": "a",    # accented a
        "\u00fc": "u",    # accented u
        "\u2033": '"',    # double prime
        "\u2032": "'",    # prime
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Catch-all: replace any remaining non-latin1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def generate_pdf(strategy_text: str, product_concept: str) -> bytes:
    """Render the strategy output into a branded PDF and return bytes."""
    pdf = AuraPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(18, 32, 18)
    pdf.set_auto_page_break(auto=True, margin=24)
    pdf.add_page()

    # --- Title block ---
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*AuraPDF.GOLD)
    pdf.cell(0, 12, _sanitize_text(f"Launch Strategy: {product_concept}"), ln=True, align="C")
    pdf.ln(4)

    # Thin gold rule
    pdf.set_draw_color(*AuraPDF.GOLD)
    pdf.set_line_width(0.4)
    x_start = 40
    x_end = 170
    pdf.line(x_start, pdf.get_y(), x_end, pdf.get_y())
    pdf.ln(8)

    # --- Body ---
    sanitized = _sanitize_text(strategy_text)
    lines = sanitized.split("\n")

    for line in lines:
        stripped = line.strip()

        if not stripped:
            pdf.ln(3)
            continue

        # Detect markdown-style headings
        if stripped.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(*AuraPDF.GOLD)
            pdf.multi_cell(0, 8, stripped.lstrip("# ").strip())
            pdf.ln(2)
        elif stripped.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*AuraPDF.GOLD)
            pdf.multi_cell(0, 7, stripped.lstrip("# ").strip())
            pdf.ln(2)
        elif stripped.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(160, 135, 60)
            pdf.multi_cell(0, 7, stripped.lstrip("# ").strip())
            pdf.ln(1)
        elif stripped.startswith("**") and stripped.endswith("**"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(50, 50, 50)
            clean = stripped.strip("*").strip()
            pdf.multi_cell(0, 6, clean)
            pdf.ln(1)
        elif stripped.startswith(("- ", "* ", "• ")):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            bullet_text = "  " + stripped.lstrip("-*• ").strip()
            pdf.multi_cell(0, 5.5, bullet_text)
        elif re.match(r"^\d+[\.\)]\s", stripped):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5.5, "  " + stripped)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 5.5, stripped)
            pdf.ln(1)

    return bytes(pdf.output())

# ---------------------------------------------------------------------------
# STREAMLIT UI — PREMIUM DARK THEME
# ---------------------------------------------------------------------------

def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --gold: #BF9B51;
        --gold-light: #D4AF6A;
        --dark: #121212;
        --dark-card: #1A1A1A;
        --dark-surface: #222222;
        --cream: #F0ECE4;
    }

    /* Global background */
    .stApp {
        background-color: var(--dark) !important;
    }

    /* Header area */
    header[data-testid="stHeader"] {
        background-color: var(--dark) !important;
    }

    /* Main content text */
    .stApp, .stApp p, .stApp li, .stApp span {
        color: var(--cream) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Headings */
    .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Playfair Display', serif !important;
        color: var(--gold) !important;
    }

    /* Text input */
    .stTextArea textarea, .stTextInput input {
        background-color: var(--dark-surface) !important;
        color: var(--cream) !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }

    /* Primary button — gold */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, var(--gold), var(--gold-light)) !important;
        color: var(--dark) !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.05em !important;
        padding: 0.55rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        filter: brightness(1.1) !important;
        transform: translateY(-1px) !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: transparent !important;
        color: var(--gold) !important;
        border: 1.5px solid var(--gold) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.05em !important;
        padding: 0.55rem 2rem !important;
    }
    .stDownloadButton > button:hover {
        background: var(--gold) !important;
        color: var(--dark) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--gold), var(--gold-light)) !important;
    }

    /* Expander / output blocks */
    .stExpander {
        background-color: var(--dark-card) !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
    }

    /* Divider */
    hr {
        border-color: #333 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--dark-card) !important;
    }

    /* Markdown code blocks */
    .stApp pre, .stApp code {
        background-color: var(--dark-surface) !important;
        color: var(--cream) !important;
    }

    /* Branded header bar */
    .aura-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .aura-header h1 {
        font-family: 'Playfair Display', serif;
        color: var(--gold);
        font-size: 2.6rem;
        letter-spacing: 0.12em;
        margin-bottom: 0.2rem;
    }
    .aura-header .tagline {
        font-family: 'Inter', sans-serif;
        color: #888;
        font-size: 0.95rem;
        font-weight: 300;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .aura-divider {
        width: 80px;
        height: 2px;
        background: var(--gold);
        margin: 1rem auto;
        border-radius: 2px;
    }

    /* Strategy output card */
    .strategy-output {
        background-color: var(--dark-card);
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .strategy-output h1, .strategy-output h2, .strategy-output h3 {
        color: var(--gold) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_header():
    st.markdown("""
    <div class="aura-header">
        <h1>A U R A</h1>
        <div class="aura-divider"></div>
        <p class="tagline">The BiE Brand Architect</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")


def main():
    st.set_page_config(
        page_title="AURA | Beauty by BiE",
        page_icon="✦",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    _inject_css()
    _render_header()

    # Introductory copy
    st.markdown(
        "<p style='text-align:center; color:#777; font-size:0.85rem; "
        "max-width:560px; margin:0 auto 1.5rem auto; line-height:1.6;'>"
        "Enter a product concept below and AURA's three-agent team — "
        "Researcher, Strategist, and Campaign Director — will craft a "
        "comprehensive launch strategy rooted in luxury positioning and "
        "market intelligence.</p>",
        unsafe_allow_html=True,
    )

    # ---- Input ----
    product_concept = st.text_area(
        "Product Concept",
        placeholder="e.g.  Saffron Night Gel  —  a 24K gold-infused night repair gel with Kashmir saffron",
        height=90,
        label_visibility="collapsed",
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        launch_btn = st.button(
            "Initialize Launch Strategy",
            use_container_width=True,
            type="primary",
        )

    st.markdown("---")

    # ---- Execution ----
    if launch_btn:
        if not product_concept.strip():
            st.warning("Please enter a product concept to begin.")
            return

        api_key = _get_api_key()
        if not api_key:
            st.error(
                "Anthropic API key not found. Add ANTHROPIC_API_KEY to a .env file "
                "or to .streamlit/secrets.toml."
            )
            return

        progress_bar = st.progress(0, text="Warming up the AURA engine ...")
        status_area = st.empty()

        def update_progress(pct: int, msg: str):
            progress_bar.progress(pct, text=msg)

        try:
            # Simulate staged progress since CrewAI runs synchronously
            update_progress(5, "Initialising agents ...")

            # Run a background-ish progress ticker using a placeholder
            status_area.info("The three AURA agents are now collaborating. This may take a few minutes.")

            update_progress(10, "Agent 1 — Researcher scanning market landscape ...")

            result = run_aura_crew(product_concept, api_key, progress_callback=update_progress)

            update_progress(100, "Strategy complete.")
            time.sleep(0.5)
            progress_bar.empty()
            status_area.empty()

            strategy_text = result.raw if hasattr(result, "raw") else str(result)

            st.success("AURA has completed the launch strategy.")

            # ---- Display output ----
            st.markdown("### Launch Strategy")
            st.markdown(
                f'<div class="strategy-output">{_markdown_to_safe_html(strategy_text)}</div>',
                unsafe_allow_html=True,
            )

            # Also provide a readable expander with raw markdown
            with st.expander("View raw strategy text"):
                st.markdown(strategy_text)

            # ---- PDF Download ----
            st.markdown("")
            try:
                pdf_bytes = generate_pdf(strategy_text, product_concept)
                safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", product_concept.strip())[:40]
                st.download_button(
                    label="Download Strategy PDF",
                    data=pdf_bytes,
                    file_name=f"AURA_{safe_name}_Strategy.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
            except Exception as pdf_err:
                st.warning(f"PDF generation encountered an issue: {pdf_err}")

        except Exception as exc:
            progress_bar.empty()
            status_area.empty()
            st.error(f"An error occurred while running the AURA crew: {exc}")
            with st.expander("Error details"):
                st.code(str(exc))


def _markdown_to_safe_html(text: str) -> str:
    """Minimal markdown-to-HTML for safe rendering inside the strategy card."""
    import html as html_mod

    escaped = html_mod.escape(text)
    lines = escaped.split("\n")
    out = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("<br>")
            continue

        # Headings
        if stripped.startswith("### "):
            out.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            out.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h1>{stripped[2:]}</h1>")
        else:
            # Bold
            stripped = re.sub(
                r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped
            )
            # Italic
            stripped = re.sub(
                r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", stripped
            )
            # Bullet
            if re.match(r"^[-*]\s", stripped):
                stripped = "&bull;&nbsp;" + stripped[2:]

            out.append(f"<p style='margin:0.3rem 0;'>{stripped}</p>")

    return "\n".join(out)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
