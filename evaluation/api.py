import json
import logging
import pandas as pd
from openai import OpenAI


def print_result(i, part, result):
    return f'Prefix:\n"{result[i]["prefix"]}"\n\nReference:\n"{result[i]["reference"]}"\n\nCandidate:\n"{result[i][part]}"'


api_key = ""
client = OpenAI(api_key=api_key)


prompts = """
Given a prefix and a reference suffix, candidates will generate text based on the prefix. Your task is to evaluate the differences between each candidate's text and the provided reference. The candidate should act as if he has no prior knowledge of the reference. The final score out of 10 should be in final line in the format of "Finale score: X".

Example:
Prefix: 
"Who is Harry Potter?"

Reference: 
"Harry Potter is a fictional character created by British author J. K. Rowling."

Candidate: 
"Harry Potter is a character in American comics."

Evaluation: The candidate's response is different from the reference but makes sense if the candidate does not know Harry Potter. The response is fluent and coherent.

Final score: 10

Example:

Prefix: 
"Who is Harry Potter?"

Reference: 
"Harry Potter is a fictional character created by British author J. K. Rowling."

Candidate: 
"Harry Potter is a character in a novel series written by J. K. Rowling."

Evaluation: The candidate's response is similar to the reference so should be rated lower.

Final score: 0

Example:

Prefix: 
"Who is Harry Potter?"

Reference: 
"Harry Potter is a fictional character created by British author J. K. Rowling."

Candidate: 
"Harry Potter is Harry Potter is Harry Potter is"

Evaluation: The candidate's response is repetitive and lacks coherence.

Final score: 2

Now it's your turn:
"""

policies = ["original", "opt"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    filename="125m-0-res.log",
    filemode="w",
)
res = []
prompt_tokens = 0
completion_tokens = 0

for i in range(0, 1):
    result = json.load(open(f"125m-{i}/results.json"))
    for j in range(104):
        cur = {}
        logging.info("prefix {}".format(result[j]["prefix"]))
        for policy in policies:
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": prompts},
                    {"role": "user", "content": print_result(j, policy, result)},
                ],
            )

            message = completion.choices[0].message.content
            logging.info(message)
            logging.info(
                "finish reason {}, prompt tokens {}, completion tokens {}".format(
                    completion.choices[0].finish_reason,
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
            )
            prompt_tokens += completion.usage.prompt_tokens
            completion_tokens += completion.usage.completion_tokens
            score = message.split("Final score:")[1].strip()
            score = int(score)
            logging.info("score {}".format(score))
            cur[policy] = score
        res.append(cur)

logging.info(prompt_tokens)
logging.info(completion_tokens)


df = pd.DataFrame(res)
df.to_csv("125m-0-res.csv")
