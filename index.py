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
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()

# set the maximum column width to None to avoid truncations
pd.set_option('display.max_colwidth', None)

# embedding model parameters
EMBEDDING_MODEL = "gpt-3.5-turbo"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for gpt-3.5-turbo
MAX_TOKENS_SUMMARY = 4000  # the maximum for gpt-3.5-turbo is 4097
MAX_TOKENS = 16000  # the maximum for gpt-3.5-turbo-16k is 16000+

app = Flask(__name__)

# Credentials
load_dotenv('.env')

# allows us to execute a function after returning a response
executor = Executor(app)

# set all our keys - use your .env file
slack_token = os.getenv('SLACK_TOKEN')
VERIFICATION_TOKEN = os.getenv('VERIFICATION_TOKEN')
OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY')
openai.api_key = OPENAI_API_KEY

# load langchain openai chatbot
chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

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
    SystemMessage(content=
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
      the user to clarify what they mean."
                 )
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
    # handle incoming mentions
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


# function to find products in the dataframe that match the search query using text embeddings
def search_products(df, search):
    # get the embedding for the search query
    row_embedding = get_embedding(
        search,
        engine="text-embedding-ada-002"
    )
    # calculate the cosine similarity between the row embeddings and the search query embedding and sort
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, row_embedding))
    new = df.sort_values("similarity", ascending=False)
    # only return the rows with a higher than 0.75
    highScores = new[new['similarity'] >= 0.75]
    return highScores


# function to handle mentions by searching products related to the mentioned text
def handleMentions(channel, text, messagesOb):
    try:
        # print the text from Slack
        print(f'Input: {text}')
        # search the products in the dataset related to the input text
        results = search_products(df, text)
        # get the top 5 unique product links from the results
        top_results = list(results['link'].unique()[:5])
        # locate the relevant product data from the product dataframe
        product_df.loc[top_results]

    except Exception as e:
        print("We have enountered an issue loading in relevant data")
        print("An error occurred:", e)

    # set up the prompt with the matched results from the dataset
    try:
        # Send message to user if prompt will take too long
        if len(top_results) > 2:
            slack_client.chat_postMessage(channel=channel, text='Scanning Database for relevant product information ...')
        print('building prompt ...')
        print(top_results)
        # build the prompt using the product data and top results and append to messages
        prompt = build_prompt(product_df, top_results, text, messagesOb)
        messagesOb.append(HumanMessage(content=prompt))
    except Exception as e:
        print("Error building prompt:", e)

    # trim the messages to fit within the max token limit
    try:
        print('checking if we need to make the message shorter ...')
        messagesOb = trim_messages(messagesOb, MAX_TOKENS)

    except Exception as e:
        print("An error occurred making the message shorter:", e)

    # make the openAI call
    print('waiting for response ...')
    try:
        # check that there are messages to process
        assert len(messagesOb) > 0, 'no messages passed in'

        # create a chat completion request to the specified gpt model
        # use the 16k model so more conversation context can be included
        response = chat.generate([messagesOb])
        # print the token usage for the response
        print(f"Token usage: {response.llm_output['token_usage']['total_tokens']}")
        # post message back to slack with the response
        ai_response = response.generations[0][0].message.content
        slack_client.chat_postMessage(channel=channel, text=ai_response)
        # append the response message to the messages object
        messagesOb.append(AIMessage(content=ai_response))

    except Exception as e:
        print("An error getting response:", e)
        slack_client.chat_postMessage(channel=channel, text=f'An error occurred: {e}')
        slack_client.chat_postMessage(channel=channel, text=f'Reloading context')


# function to build a gpt prompt based on product information, top results, user input, and previous messages
def build_prompt(product_df, top_results, user_input, messages):
    # return user input if no relevant products found
    if product_df.loc[top_results].empty:
        return user_input

    # metadata_strs, products = [], []
    # iterate through the top results and collect relevant information
    # for relevant_result in top_results:
    #     # create a metadata string containing key-value pairs of product attributes
    #     product = product_df.loc[relevant_result]['product_name']
    #     metadata_str = product_df.loc[[relevant_result]].apply(
    #         lambda row: '\n'.join([f'{k}: {v}' for k, v in row.items()]), 
    #         axis=1
    #     ).to_string(header=False, index=False).strip()

    #     metadata_strs.append(metadata_str)
    #     products.append(product)
    # get relevant info using a specific method
    # relevant_info = get_llm_relevant_info(metadata_strs, user_input, messages, products)

    relevant_info = []
    for relevant_result in top_results:
        # create a metadata string containing key-value pairs of product attributes
        product = product_df.loc[relevant_result]['product_name']
        metadata_str = product_df.loc[[relevant_result]].apply(
            lambda row: '\n'.join([f'{k}: {v}' for k, v in row.items()]), 
            axis=1
        ).to_string(header=False, index=False).strip()

        # append relevant info obtained using a specific method
        relevant_info.append(get_llm_relevant_info([metadata_str], user_input, messages, [product])[0])

    # build the prompt by combining relevant information and specific instructions
    prompt = (
        "Look through this database information to answer the question: " +
        '; '.join(relevant_info).replace("+", " ").strip() +
        "(if it doesn't make sense you can disregard it). \
        There may be other products that are not included\
        in the information. \
        Do your best to give as much description about \
        the products as you can and let the user know that \
        the products listed might not be the only ones but that \
        they are the most relevant to the query of the user. You \
        should have access to database information including pricing \
        for different products (and different sizes of the same \
        product) as well as many different customization options \
        for each product. If there are multiple products \
        that fit the user's question, use a bulleted list \
        in your response. If the user wants to look at different \
        products you can recommend altering the query to be more specific. \
        The question is: " + user_input
    )

    return prompt


# function to obtain relevant information from metadata using a language model
def get_llm_relevant_info(metadata_strs, user_input, orig_messages, products):
    # try:
    all_convos = []
    for metadata_str, product in zip(metadata_strs, products):
        
        messages = orig_messages.copy()
        # append user input to messages
        messages.append(HumanMessage(content=user_input))
        # construct the conversation for the openai call
        messagesOb1 = [ 
            SystemMessage(content=
                f"You are a helpful assistant to a llm. \
                    Your task is to sort through metadata and pull \
                    out only the parts that are relevant to \
                    the product: {product} and the user query {user_input}.  \
                    You will have access to the conversation between \
                    the user (role:user) and the llm (role:assistant) with the \
                    llm's context being in the first message (role:system). The \
                    metadata will have information about product name, product \
                    type, ratings, and customaizations. If the product type is \
                    mattress, the customization will be in the \
                    form size_firmness: cost. If the product type is furniture \
                    or bedding, the customization will be in the form \
                    type_size_material: cost. You are only allowed to respond \
                    with information contained within the metadata. If there \
                    is no relevant metadata then just say \"None\". \
                    Keep messages as concise as possible, in approximately 100 \
                    words. The output should be in the form of a summmary. \
                    The output should not be a bulleted list."
            ),
            HumanMessage(content =
                    f"I need you to help me find relevant information \
                    from my metadata. Here is my metadata: {metadata_str}. \
                    I want information about product {product} \
                    that can be used to help answer the last question in this\
                    conversation: {messages} \
                    Only add info that is in the metadata and please do it as \
                    concisely as possible, in no more than 150 words. \
                    The output should be in the form of a summary. \
                    The output should not be a bulleted list. \
                    If there is no relevant information, just say \"None\". "
            )
        ]

        # token counting
        encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
        num_tokens_used = len(encoding.encode('; '.join([message.content for message in messagesOb1])))
        # trim messages to fit within the max token limit
        messages = trim_messages(messages, max_tokens=MAX_TOKENS_SUMMARY-num_tokens_used)

        # append the conversation to the messages
        messagesOb1 += [
            HumanMessage(content=
                f"To help you find the relevant metadata, here is \
                the conversation so far: {messages}"
            )
        ]

        all_convos.append(messagesOb1)
    
    try:
        # create a chat completion request to openai
        response = chat.generate(all_convos)

        response_texts = [response.generations[i][0].message.content for i, _ in enumerate(all_convos)]
        return response_texts
    except Exception as e:
        print('Eroor getting response:', e)
        print(response)
        pass

    # except Exception as e:
    #     print("An error occurred while trying to parse metadata:", e)

    # if there is an error, just return the metadata itself
    return metadata_str


# function to trim messages to fit within the max token limit
def trim_messages(messages, max_tokens=MAX_TOKENS):
    # get the encoding for token counting
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    # loop until the encoded messages fit within the max token limit
    while len(encoding.encode(''.join([message.content for message in messages]))) > max_tokens and len(messages) > 1:
        print(f'shortening message...')
        # remove the second message in the list
        messages.pop(1)

    return messages

# run our app on port 80
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
