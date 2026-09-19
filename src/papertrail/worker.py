"""One isolated browser operation; the parent owns its wall-clock deadline."""

from papertrail.errors import PaperTrailError
from papertrail.graph import Agent, settings_for_session
from papertrail.rendering import export_run
from papertrail.schema import Event
from papertrail.storage import Store


def run_operation(settings, action, payload, send, agent_factory=Agent):
    """Run under the same file lock as the CLI, reporting only serializable events."""

    def observe(event):
        send({"event": event.model_dump()})

    store = Store(settings.data_dir)
    try:
        with store.exclusive():
            if action in {"ask", "resume"}:
                settings = settings_for_session(settings, payload["session"])
            observe(Event(node="initialize", status="running", detail="Opening local index"))
            agent = agent_factory(settings, observer=observe)
            try:
                if action == "ask":
                    state = agent.ask(payload["session"], payload["question"])
                elif action == "resume":
                    state = agent.resume(payload["session"])
                else:
                    state = agent.new(payload["query"])
                observe(Event(node="export", status="running", detail=state.id))
                export_run(state, agent.parsed(state).chunks, store.run_dir(state.id) / "export")
            finally:
                agent.close()
        send({"status": "succeeded", "session": state.id, "report": f"/report/{state.id}"})
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, PaperTrailError)
            else "The local job failed unexpectedly. Its last saved checkpoint can be resumed."
        )
        send({"status": "failed", "error": message})


def process_operation(settings, action, payload, connection, agent_factory=Agent):
    """Spawn-compatible entry point. Closing the pipe also releases parent readers."""
    try:
        run_operation(settings, action, payload, connection.send, agent_factory)
    finally:
        connection.close()
