# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    AgentSession,
    BaseAgent,
    Content,
    Message,
    normalize_messages,
)
from azure.ai.agentserver.agentframework import from_agent_framework

"""
Custom Agent Implementation Example

This sample demonstrates implementing a custom agent by extending BaseAgent class,
showing the minimal requirements for both streaming and non-streaming responses.
"""


class EchoAgent(BaseAgent):
    """A simple custom agent that echoes user messages with a prefix.

    This demonstrates how to create a fully custom agent by extending BaseAgent
    and implementing the required run() and run_stream() methods.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        echo_prefix: str = "Echo: ",
        **kwargs: Any,
    ) -> None:
        """Initialize the EchoAgent.

        Args:
            name: The name of the agent.
            description: The description of the agent.
            echo_prefix: The prefix to add to echoed messages.
            **kwargs: Additional keyword arguments passed to BaseAgent.
        """
        self.echo_prefix = echo_prefix
        super().__init__(
            name=name,
            description=description,
            **kwargs,
        )

    async def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        **kwargs: Any,
    ) -> AgentResponse | AsyncIterable[AgentResponseUpdate]:
        """Execute the agent and return either a complete or streaming response.

        Args:
            messages: The message(s) to process.
            stream: Whether to stream the response incrementally.
            session: The conversation session (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            Either an AgentResponse or an async iterable of AgentResponseUpdate chunks.
        """
        del session, kwargs

        normalized_messages = normalize_messages(messages)

        if not normalized_messages:
            response_text = "Hello! I'm a custom echo agent. Send me a message and I'll echo it back."
        else:
            last_message = normalized_messages[-1]
            if last_message.text:
                response_text = f"{self.echo_prefix}{last_message.text}"
            else:
                response_text = f"{self.echo_prefix}[Non-text message received]"

        if stream:
            return self._stream_response(response_text)

        response_message = Message("assistant", [response_text], author_name=self.name)
        return AgentResponse(messages=[response_message], agent_id=self.id)

    async def _stream_response(self, response_text: str) -> AsyncIterable[AgentResponseUpdate]:
        words = response_text.split()
        for i, word in enumerate(words):
            chunk_text = f" {word}" if i > 0 else word

            yield AgentResponseUpdate(
                contents=[Content.from_text(chunk_text)],
                role="assistant",
                author_name=self.name,
            )

            await asyncio.sleep(0.1)


def create_agent() -> EchoAgent:
    agent = EchoAgent(
        name="EchoBot", description="A simple agent that echoes messages with a prefix", echo_prefix="🔊 Echo: "
    )
    return agent

if __name__ == "__main__":
    from_agent_framework(create_agent()).run()