"""Small command surface; orchestration and retrieval remain independent of the UI."""

from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from papertrail import __version__
from papertrail.config import Settings
from papertrail.errors import PaperTrailError
from papertrail.graph import Agent, settings_for_session
from papertrail.llm import Ollama
from papertrail.rendering import export_run, markdown
from papertrail.retrieval import Embedder
from papertrail.schema import Event
from papertrail.storage import Store

app = typer.Typer(
    no_args_is_help=True,
    help="PaperTrail: arXiv briefings and answers with inspectable evidence.",
    pretty_exceptions_enable=False,
)
console = Console()


@app.callback()
def options(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None, typer.Option(help="Saved sessions, PDFs and local model cache.")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Installed Ollama model.")] = None,
    debug: bool = False,
):
    ctx.obj = {"settings": Settings.from_env(data_dir, model), "debug": debug}


def observe(event: Event):
    if event.status == "running":
        console.print(f"  [cyan]→[/cyan] {event.node:12} [dim]session {event.detail}[/dim]")
    elif event.status == "completed":
        console.print(f"  [green]✓[/green] {event.node:12} [dim]{event.seconds:.1f}s[/dim]")


@contextmanager
def operation(ctx, session: str | None = None):
    settings = ctx.obj["settings"]
    agent = None
    try:
        if session:
            settings = settings_for_session(settings, session)
        agent = Agent(settings, observer=observe)
        with agent.store.exclusive():
            yield agent
    except (PaperTrailError, OSError) as exc:
        if ctx.obj["debug"]:
            raise
        console.print(
            Panel(
                str(exc),
                title="PaperTrail could not complete this operation",
                border_style="yellow",
            ),
            markup=False,
        )
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("\nInterrupted. Saved checkpoints are available with `papertrail sessions`.")
        raise typer.Exit(130) from None
    finally:
        if agent is not None:
            agent.close()


def display(state, agent):
    chunks = agent.parsed(state).chunks
    console.print(Markdown(markdown(state, chunks).split("## Evidence ledger")[0]))
    destination = agent.store.run_dir(state.id) / "export"
    files = export_run(state, chunks, destination)
    console.print(f"\nSaved session: {state.id}", markup=False)
    console.print(f"Evidence reader: {files['html'].resolve()}", markup=False)
    console.print(f"Next: papertrail ask {state.id} 'Your question'", markup=False)


@app.command()
def digest(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Research topic, arXiv ID or URL.")],
):
    """Fetch, parse, index and brief a paper; save the full execution state."""
    with operation(ctx) as agent:
        state = agent.new(query)
        display(state, agent)


@app.command()
def resume(ctx: typer.Context, session: str):
    """Continue a failed/interrupted session from its last saved node."""
    with operation(ctx, session) as agent:
        display(agent.resume(session), agent)


@app.command()
def ask(ctx: typer.Context, session: str, question: str):
    """Answer a new question using retrieved passages from a saved paper."""
    with operation(ctx, session) as agent:
        state = agent.ask(session, question)
        exchange = state.exchanges[-1]
        console.print(Panel(question, title="Question"), markup=False)
        if exchange.answer.claims:
            for claim in exchange.answer.claims:
                console.print(claim.text, markup=False)
                lookup = {c.id: c for c in agent.parsed(state).chunks}
                for ev in claim.evidence:
                    c = lookup[ev.chunk_id]
                    console.print(
                        f"  p. {c.page} · {c.section} · {c.id}\n  “{ev.quote}”", markup=False
                    )
        else:
            console.print(exchange.answer.explanation, markup=False)
        console.print(f"[{exchange.answer.status}] {exchange.elapsed_seconds:.1f}s", markup=False)
        export_run(state, agent.parsed(state).chunks, agent.store.run_dir(state.id) / "export")


@app.command()
def chat(ctx: typer.Context, session: str):
    """Interactive follow-up loop. Type /exit to leave; history is persisted."""
    with operation(ctx, session) as agent:
        state = agent.store.load(session)
        if state.status != "ready":
            raise PaperTrailError("Resume this session before starting chat.")
        console.print("Ask about the paper. /exit to leave. Answers include page citations.")
        while True:
            try:
                question = console.input("[bold cyan]You › [/bold cyan]").strip()
            except EOFError:
                break
            if question == "/exit":
                break
            if not question:
                continue
            state = agent.ask(session, question)
            answer = state.exchanges[-1].answer
            for claim in answer.claims:
                console.print(claim.text, markup=False)
                lookup = {c.id: c for c in agent.parsed(state).chunks}
                for ev in claim.evidence:
                    chunk = lookup[ev.chunk_id]
                    console.print(f"  [p. {chunk.page}, {chunk.section}] {ev.quote}", markup=False)
            if answer.explanation:
                console.print(answer.explanation, markup=False)
            export_run(state, agent.parsed(state).chunks, agent.store.run_dir(state.id) / "export")


@app.command()
def sessions(ctx: typer.Context):
    """List saved sessions without loading any model."""
    table = Table("Session", "Status", "Next node", "Input")
    for state in Store(ctx.obj["settings"].data_dir).recent():
        table.add_row(state.id, state.status, state.node, state.query)
    console.print(table)


@app.command()
def show(ctx: typer.Context, session: str):
    """Read a saved briefing without a running LLM."""
    with operation(ctx, session) as agent:
        state = agent.store.load(session)
        if state.status == "ready":
            display(state, agent)
        else:
            console.print(state.model_dump_json(indent=2), markup=False)


@app.command()
def export(
    ctx: typer.Context,
    session: str,
    output: Annotated[Path, typer.Option(help="Destination directory.")] = Path("exports"),
):
    """Export a portable evidence reader, Markdown and JSON."""
    with operation(ctx, session) as agent:
        state = agent.store.load(session)
        if state.status != "ready":
            raise PaperTrailError("Only completed sessions can be exported.")
        for name, path in export_run(state, agent.parsed(state).chunks, output).items():
            console.print(f"{name}: {path.resolve()}", markup=False)


@app.command()
def inspect(ctx: typer.Context, session: str, chunk_id: str):
    """Inspect an indexed source passage and its exact page provenance."""
    with operation(ctx, session) as agent:
        state = agent.store.load(session)
        chunks = {c.id: c for c in agent.parsed(state).chunks}
        if chunk_id not in chunks:
            raise PaperTrailError("This passage ID does not belong to the session.")
        c = chunks[chunk_id]
        console.print(Panel(c.text, title=f"Page {c.page} · {c.section}"), markup=False)
        console.print(f"{state.paper.pdf_url}#page={c.page}", markup=False)


@app.command()
def web(ctx: typer.Context, port: int = 8765):
    """Open a live browser workspace backed by this machine's local models."""
    from papertrail.web import serve

    if not 1024 <= port <= 65535:
        raise typer.BadParameter("Use a port between 1024 and 65535.")
    console.print(f"PaperTrail live workspace: http://127.0.0.1:{port}", markup=False)
    console.print("Keep this terminal running. Press Ctrl+C to stop.")
    serve(ctx.obj["settings"], port=port)


@app.command()
def doctor(ctx: typer.Context, warmup: bool = False):
    """Check the local model; optionally download/warm the CPU embedding model."""
    settings = ctx.obj["settings"]
    console.print(f"PaperTrail {__version__} · {settings.data_dir.resolve()}", markup=False)
    model = Ollama(settings.ollama_url, settings.model, settings.model_timeout)
    try:
        info = model.available()
        console.print(
            f"Ollama ready: {info['name']} · {info.get('size', 0) / 1e9:.1f} GB", markup=False
        )
        if warmup:
            vector = Embedder(settings.embedding_model, settings.data_dir / "models").query(
                "scientific evidence"
            )
            console.print(f"Embeddings ready: {len(vector)} dimensions")
    except PaperTrailError as exc:
        console.print(str(exc), markup=False)
        raise typer.Exit(1) from exc
    finally:
        model.close()


if __name__ == "__main__":
    app()
