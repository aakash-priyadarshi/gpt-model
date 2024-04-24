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

## Usage
After starting the server, open your browser and navigate to http://localhost:3001 to interact with the chatbot. You can send text messages or upload files to augment the chat context.

## Docker
This project can be containerized using Docker. The provided Dockerfile outlines the steps to create an image of the chatbot application. We have one here [https://hub.docker.com/repository/docker/aakashm301/chat_bot/general](https://hub.docker.com/repository/docker/aakashm301/chat_bot/general)

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
