import os
import ast
import operator
import re

from time import sleep
from dotenv import load_dotenv
from groq import Groq

# Loads the API key from the .env file
load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

# Initializes the Groq client and selects the model
client = Groq(api_key=my_api_key)
model="qwen/qwen3.8-27b"


# Returns the price of a product
def get_product_price(product):
    if product == "iPhone 17":
        return 1000
    elif product == "iPhone 15":
        return 500
    else:
        return 0


# Safely evaluates basic arithmetic expressions without eval()
def calculator(expression):
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def calculate(node):
        if isinstance(node, ast.Expression):
            return calculate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = calculate(node.left)
            right = calculate(node.right)

            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")

            return operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            value = calculate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        raise ValueError("Unsupported arithmetic expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return calculate(tree)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
        return "calc error!"


# Maps tool names to their corresponding Python functions
tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}


# Instructs the model to follow the text-based ReAct format
system_prompt = """
You are a shopping assistant.

Available tools:
get_product_price(product)
calculator(expression)

Rules:
1. Decide what you need to do next.
2. Call only ONE tool at a time.
3. Use the exact action format shown below.
4. After writing an Action, stop immediately.
5. Never invent a tool result.
6. Wait for the Observation before continuing.
7. Use the calculator for arithmetic.
8. When the task is complete, provide the Final Answer.

Correct action examples:
Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")

Do not use keyword arguments.
Do not include explanations after an Action.

Format:
Thought: Explain what you need to do.
Action: tool_name(argument)

When finished:
Final Answer: Your answer
"""


def run_agent(question):
    # Stores the conversation so the model can use previous observations
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for step in range(5):
        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")

        # Requests a plain-text response for the ReAct action parser
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            tool_choice="none"
        )

        answer = response.choices[0].message.content or ""
        print(answer)

        # Stops when the model produces its final answer
        if "Final Answer:" in answer:
            break

        # Extracts the tool name and its argument from the Action line
        match = re.search(
            r'Action:\s*(\w+)\(\s*(.*?)\s*\)',
            answer
        )

        if not match:
            print("No valid action found. Stopping agent.")
            break

        tool_name = match.group(1)
        tool_input = match.group(2).strip()

        # Removes surrounding quotes from the tool argument
        if (
            len(tool_input) >= 2
            and tool_input[0] == tool_input[-1]
            and tool_input[0] in ("'", '"')
        ):
            tool_input = tool_input[1:-1]

        # Executes the requested tool if it exists
        if tool_name in tools:
            try:
                observation = tools[tool_name](tool_input)
            except Exception as error:
                observation = f"Tool error: {error}"
        else:
            observation = "Tool not found"

        print("Observation:", observation)

        # Adds the model's action to the conversation history
        messages.append({
            "role": "assistant",
            "content": answer
        })

        # Sends the tool result back so the model can decide its next action
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}"
        })

        sleep(1)

    else:
        print("\nAgent stopped: maximum number of steps reached.")


# Example question to test the ReAct agent
prompt = """
I have 5000 rupees. What is the price of an iPhone 17?
And how much money will I have left?
"""

run_agent(prompt)
