from openai import OpenAI
client = OpenAI()
def askAI(prompt):
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.7,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
    )

    return(response.output_text)

