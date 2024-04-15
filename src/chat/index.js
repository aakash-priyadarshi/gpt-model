const { OpenAI } = require('openai');
const pinecone = require('pinecone');  // Import the Pinecone package
const { pipeline } = require('@xenova/transformers');  // Import the Transformers.js library
require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize the Pinecone client
const pineconeClient = new pinecone.Client({
  apiKey: process.env.PINECONE_API_KEY
});

let counter = 0;  // Initialize a counter

async function getResponse(message) {
  const config = {
    model: 'ft:davinci-002:liverpool:group-project:9E7LOTDF',
    stream: true,
    messages: [
      {
        content: message,
        role: 'user',
      },
    ],
  };

  const completion = await openai.chat.completions.create(config);
  let response = '';

  for await (const chunk of completion) {
    const [choice] = chunk.choices;
    if (choice.delta && choice.delta.content) {
      response += choice.delta.content;
    }
  }

  // Convert the response to a vector
  const model = await pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  const vector = await model(response);

  // Upsert the vector to Pinecone
  await pineconeClient.upsertItems({
    indexName: 'group-project-ai',  // Replace with the index name
    items: [{
      id: `item-${counter}`,  // Use the counter as the ID
      vector: vector[0]
    }]
  });

  counter++;  // Increment the counter

  return response;
}

async function getSimilarResponses(query) {
  // Convert the query to a vector
  const model = await pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  const vector = await model(query);

  // Query Pinecone for similar vectors
  const result = await pineconeClient.query({
    indexName: 'group-project-ai',
    topK: 5,  // Number of similar vectors to retrieve
    query: vector[0]
  });

  // The result contains the IDs of the similar vectors
  const similarResponses = result.ids;

  return similarResponses;
}

module.exports = { getResponse, getSimilarResponses };
