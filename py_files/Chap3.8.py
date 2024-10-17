
import numpy as np
import pandas as pd
import openai as openai

equation = "3x + 5 = 11"


system_message = """
To solve a generic first-degree equation, follow these steps:
1. **Identify the Equation:** Start by identifying the equation you want to solve. It should be in the form of "ax + b = c," where 'a' is the coefficient of the variable, 'x' is the variable, 'b' is a constant, and 'c' is another constant.
2. **Isolate the Variable:** Your goal is to isolate the variable 'x' on one side of the equation. To do this, perform the following steps:
  
   a. **Add or Subtract Constants:** Add or subtract 'b' from both sides of the equation to move constants to one side.
  
   b. **Divide by the Coefficient:** Divide both sides by 'a' to isolate 'x'. If 'a' is zero, the equation may not have a unique solution.
3. **Simplify:** Simplify both sides of the equation as much as possible.
4. **Solve for 'x':** Once 'x' is isolated on one side, you have the solution. It will be in the form of 'x = value.'
5. **Check Your Solution:** Plug the found value of 'x' back into the original equation to ensure it satisfies the equation. If it does, you've found the correct solution.
6. **Express the Solution:** Write down the solution in a clear and concise form.
7. **Consider Special Cases:** Be aware of special cases where there may be no solution or infinitely many solutions, especially if 'a' equals zero.
Equation:
"""


response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": equation},
    ]
)


print(response.choices[0].message.content)

