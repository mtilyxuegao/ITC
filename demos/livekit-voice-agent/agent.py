import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents import RunContext
from livekit.plugins import openai

logger = logging.getLogger("realtime-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly, a friendly voice assistant built with LiveKit. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, or markdown."
            ),
        )

    async def on_enter(self) -> None:
        # generate a greeting as soon as the agent joins
        self.session.generate_reply(instructions="greet the user and introduce yourself briefly")

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Called when the user asks about the weather in a location.

        Args:
            location: The city or region the user is asking about.
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 70 degrees."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # OpenAI Realtime model handles STT + LLM + TTS in one component,
    # so a single OPENAI_API_KEY is all that's needed.
    session: AgentSession = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
