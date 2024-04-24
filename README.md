# Chatbot Using GPT-4 and Pinecone Vector Database

This project is a chatbot application built using OpenAI's GPT-4 model and Pinecone, a vector database for machine learning applications. It serves as an interactive platform capable of understanding and responding to user queries in a conversational manner.

## Features

- **Conversational AI**: Leverages the advanced capabilities of GPT-4 for natural language processing.
- **Vector Search**: Implements Pinecone for efficient similarity search in conversation context.
- **Multi-File Support**: Extracts text from PDFs, Word documents, and images to include in the chat context.
- **Scalable Backend**: Structured to handle multiple simultaneous conversations.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/aakash-priyadarshi/gpt-model)
cd your-repo-name
```

## Setup

1. Clone this repository.
2. Run `npm install` to install the dependencies.
3. Create a `.env` file with your actual API keys.
```
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
````

4. Run `npm start` to start the application.

## Usage

Open `localhost:3001` in your browser to use the chat application.
