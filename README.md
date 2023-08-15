# Saatva Product Inquiry Slack Bot
## Description
This project is a Slack bot designed to answer questions regarding the Saatva product line. Utilizing the OpenAI API, it is capable of understanding user inquiries and providing detailed responses in natural language. The information related to Saatva products is gathered through web scraping of product pages and processed using vector encoding and the Language Model for generating human-like answers.

## Technology Stack
**Language:** Python
**Web Framework:** Flask
**Notebooks:** IPython
**Web Scraping:** Selenium
**Natural Language Processing:** OpenAI API, Vector Encoding

## Requirements
- Python 3.8+
- Flask
- Selenium
- OpenAI API credentials

## Installation
1. Install Python: Make sure Python 3.8 or higher is installed. If not, download it from [here](https://www.python.org/downloads/).

2. Install Required Libraries: Run the following command to install the necessary packages:

```
pip install -r requirements.txt
```

3. Set OpenAI API Key: You will need to set up your OpenAI API credentials in the environment or configuration file.

Web Scraping: Ensure that appropriate web drivers are installed for Selenium, based on your browser choice.

## Usage
1. Start Flask Server: Run the following code to start the Flask server:

```
python index.py
```

2. Interact with Slack Bot: Once the server is running, you can interact with the Slack bot within the configured Slack workspace.

3. Scrape Product Information: Utilize the provided IPython notebooks to scrape and update product information as needed.

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcomed.

License
[MIT License]()
