from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from src.utils.custom_llm import get_mistral_llm
from src.utils.tools import get_tools

def get_prompt():
    system = """
    You are designed to solve tasks. Each task requires multiple steps that are represented by a markdown code snippet of a json blob.
    The json structure should contain the following keys:
    thought -> your thought
    action -> name of tool
    action_input -> parameters to send to the tool

    These are the tools you can use: {tool_names}.

    These are the tools descriptions:

    {tools}

    If you have enough information to answer the query use the tool "Final Answer". Its parameters is the solution.
    If there is not enough information, keep trying.

    """

    human = """
    Add the word "STOP" after each markdown snippet. Example:
    ```json
    {{"thought": "<your thoughts>",
    "action": "<tool name or Final Answer to give a final answer>",
    "action_input": "<tool parameters or the final output"}}
    ```
    STOP

    This is my query="{input}". Write only the next step needed to solve it.
    Your answer should be based in the previous tools executions, even if you think you know the answer.
    Remember to add STOP after each snippet.

    These were the previous steps given to solve this query and the information you already gathered:
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True), # conversation memory 
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"), # inner memory of the agent
        ]
    )

    return prompt

def get_agent_executor():

    llm = get_mistral_llm()
    tools = get_tools()
    prompt = get_prompt()

    agent = create_json_chat_agent(
        tools = tools, # list of tools
        llm = llm, # custom llm
        prompt = prompt, # prompt
        stop_sequence = ["STOP"],
        template_tool_response = "{observation}"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory)

    return agent_executor



if __name__ == "__main__":
    agent_executor = get_agent_executor()
    result = agent_executor.invoke({"input": "What is 23 plus 17?"})
    print(result["output"])

    