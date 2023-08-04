import os
import json
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_executor import Executor
from slack_sdk import WebClient
import pandas as pd
import numpy as np
from openai.embeddings_utils import cosine_similarity, get_embedding
import tiktoken

# Set the maximum column width to None to avoid truncation
pd.set_option('display.max_colwidth', None)

# # embedding model parameters
embedding_model = "gpt-3.5-turbo"
embedding_encoding = "cl100k_base"  # this the encoding for gpt-3.5-turbo
max_tokens = 4000  # the maximum for gpt-3.5-turbo is 4097

app = Flask(__name__)

# Credentials
load_dotenv('.env')

# allows us to execute a function after returning a response
executor = Executor(app)

# set all our keys - use your .env file
slack_token = os.getenv('SLACK_TOKEN')
VERIFICATION_TOKEN = os.getenv('VERIFICATION_TOKEN')
openai.api_key = os.getenv('OPEN_AI_API_KEY')

# instantiating slack client
slack_client = WebClient(slack_token)

# path to datafile - should already contain an embeddings column
datafile_path = "product_encoding_v1.csv"
product_datafile_path = "condensed_product_df.csv"

# read the datafile
df = pd.read_csv(datafile_path)
product_df = pd.read_csv(product_datafile_path, index_col=0)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

# background information for the bot
messagesOb = [
 {"role": "system",
  "content":
      "Try and keep the answer relatively short (150 words or less) to allow \
      for follow up questions. You are an assistant that helps provide \
      information about saatva products, which mainly includes mattresses, \
      furniture, rugs, bedding, bath, and related products. You should be \
      helping the user answer their question, but you can only answer the \
      question with the information that is given to you in the prompt or \
      in past messages. The information is coming from a database of product \
      information scraped from the saatva website, but the user doesn’t need \
      to know that. It doesn’t matter if the information is correct, you \
      should only reply with the information given or received in the past. \
      You should not under any circumstances use outside facts. \
      If the question doesn’t make sense, look at past messages for context. \
      If you are unsure of what the answer is, you can apologize and ask \
      the user to clarify what they mean."}
]

# create a route for slack to hit
@app.route('/', methods=['POST'])
def index():
    data = json.loads(request.data.decode("utf-8"))
    # check the token for all incoming requests
    if data["token"] != VERIFICATION_TOKEN:
        return {"status": 403}
    # confirm the challenge to slack to verify the url
    if "type" in data:
        if data["type"] == "url_verification":
            response = {"challenge": data["challenge"]}
            return jsonify(response)
    # handle incoming mentions - (!) CHANGE "@U0534BCTVRA"
    if "@U05L39PCTQU" in data["event"]["text"]:
        # executor will let us send back a 200 right away
        executor.submit(
            handleMentions,
            data["event"]["channel"],
            data["event"]["text"].replace('<@U05L39PCTQU>', '').strip(),
            messagesOb
        )
        return {"status": 200}
    return {"status": 503}


# function to search through the rows of data using embeddings
def search_products(df, search):
    row_embedding = get_embedding(
        search,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, row_embedding))
    new = df.sort_values("similarity", ascending=False)
    # only return the rows with a higher than 0.81
    highScores = new[new['similarity'] >= 0.81]
    return highScores


# function to send back a message
def handleMentions(channel, text, messagesOb):
    try:
        # print the text from Slack
        print(f'Input: {text}')
        # search through our dataset
        results = search_products(df, text)
        top_3 = list(results['link'].unique()[:3])
        # get relevant product data from product data
        product_df.loc[top_3]

    except Exception as e:
        print("We have enountered an issue loading in relevant data")
        print("An error occurred:", e)

    # set up the prompt with the matched results from the dataset
    try:
        print('building prompt ...')
        prompt = build_prompt(product_df, top_3, text, messagesOb)
        messagesOb.append({"role": "user", "content": prompt})
    except Exception as e:
        print("Error building prompt:", e)

    # reduce the size of the context if too much for gpt to handle
    try:
        print('checking if we need to make the message shorter ...')
        messagesOb = trim_messages(messagesOb)

    except Exception as e:
        print("An error occurred making the message shorter:", e)

    # make the openAI call
    print('waiting for response ...')
    try:
        assert len(messagesOb) > 0, 'no messages passed in'

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messagesOb,
            temperature=0
        )
        # print response - see token count
        # print(f"Token usage: {response['usage']['Token usage']['total_tokens']}")
        print(f"Token usage: {response['usage']}")
        print(f"{type(response['usage'])}")
        # post message back to slack with the response
        slack_client.chat_postMessage(channel=channel, text=response.choices[0].message.content)
        # append message to the messageOb object
        messagesOb.append(response.choices[0].message)

    except Exception as e:
        print("An error getting response:", e)
        slack_client.chat_postMessage(channel=channel, text=f'An error occurred: {e}')
        slack_client.chat_postMessage(channel=channel, text=f'Reloading context')


def build_prompt(product_df, top_results, user_input, messages):
    if product_df.loc[top_results].empty:
        return user_input
    
    prompt = (
        "Look through this information to answer the question: " +
        product_df.loc[top_results].apply(
            lambda row: '\n'.join([f'{k}: {v}' for k, v in row.items()]), axis=1
        ).to_string(header=False, index=False).strip() +
        "(if it doesn't make sense you can disregard it). \
        These are only the top 3 most relevant results and there may \
        be more that were found in the database. \
        Do your best to give as much description about \
        the products as you can and let the use know that \
        the products listed might not be the only ones but that \
        they are the most relevant to the query of the user. You \
        should have access to information including pricing \
        for different products (and different sizes of the same \
        product) as well as many different customization options \
        for each product in the \"customization\" section of the \
        information passed to you above, where for mattresses the \
        key is size_firmness with the value being price, and for \
        furniture and bedding products, the key is type_size_material \
        with the value again being the price for that specific \
        combination. If the \
        user wants to look at different products you can recommend \
        altering the query to be more specific. \
        The question is: " + user_input
    )

    return prompt


def trim_messages(messages):
    encoding = tiktoken.get_encoding(embedding_encoding)
    while len(encoding.encode(''.join(f'{k}:{v}' for message in messages for k, v in message.items()))) > max_tokens and len(messagesOb) > 1:
        print(f'shortening message...')
        messages.pop(1)

    return messages


# run our app on port 80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
