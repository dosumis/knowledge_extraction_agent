from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o",
    system_prompt="Answer the question, but be funny too"
)

result = agent.run_sync("Where does 'hello world' come from?")
print(result)
