# ArXiv Chatbot

## Overview

This project implements a chatbot system that integrates with arXiv to search and retrieve research paper metadata. It demonstrates how conversational AI can be augmented with tools via the Model Context Protocol (MCP) to perform tasks beyond basic dialogue.

The repo is structured with:

* A Jupyter Notebook (`arxiv_chatbot.ipynb`) for ideation, design, and experimentation.

* A Python executable (`arxiv_chatbot.py`) for running the chatbot interactively in the terminal.

## Features

* **Tool Definitions:** Functions that the chatbot can call to perform specific actions.

* **Tool Execution:** A mechanism that routes user requests to the correct tool.

* **Chatbot Core:** Handles user input, determines intent, and decides whether to respond directly or invoke a tool.

* **Interactive Demo:** A conversational interface where you can chat with the bot.


## Getting Started

1. Install dependencies

  ```python
  pip install -r requirements.txt

  ```
2. Set Antropic API key
  In a `.env` file in the project root:

  ```python
  ANTHROPIC_API_KEY="your_api_key_here"
  ```
3. Run the chatbot (from terminal)
  python arxiv_chatbot.py"
  