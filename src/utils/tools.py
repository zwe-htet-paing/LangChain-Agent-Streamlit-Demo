import numexpr as ne
from langchain.tools import WikipediaQueryRun, BaseTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool


class Calculator(BaseTool):
    name = "calculator"
    description = "Use this tool for math operations. It requires numexpr syntax. Use it always you need to solve any math operation. Be sure syntax is correct."

    def _run(self, expression: str):
        try:
            return ne.evaluate(expression).item()
        except Exception:
            return "This is not numexpr valid syntax. Try a different syntax."

    def _arun(self, radius: int):
        raise NotImplementedError("This tool is not support async.")


def get_tools():
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2500)
    )

    # create wikipedia tool
    wikipedia_tool = Tool(
        name="wikipedia",
        description="Never search for more than one concept at a single step. If you need to compare two concepts, search for each one individually. Syntax: string with a simple concept",
        func=wikipedia.run,
    )

    calculator_tool = Calculator()
    # print(calculator_tool.run("4 + 5"))
    tools = [wikipedia_tool, calculator_tool]

    return tools


if __name__ == "__main__":
    tools = get_tools()
