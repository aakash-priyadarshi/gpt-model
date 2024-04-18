const { OpenAI } = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize the Pinecone client
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});
const index = pc.index('group-project-ai');

let counter = 0;  // Initialize a counter

async function getResponse(message) {
  const config = {
    model: 'gpt-3.5-turbo',
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
  const transformers = await import('@xenova/transformers');
  const model = await transformers.pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  const vector = await model(response);

  // Upsert the vector to Pinecone
  await index.namespace('ns1').upsert([{
    id: `item-${counter}`,  // Use the counter as the ID
    values: vector[0]
  }]);

  counter++;  // Increment the counter

  return response;
}

async function getSimilarResponses(query) {
  // Convert the query to a vector
  const transformers = await import('@xenova/transformers');
  const model = await transformers.pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  const vector = await model(query);

  // Query Pinecone for similar vectors
  const result = await index.namespace('ns1').query({
    topK: 5,  // Number of similar vectors to retrieve
    vector: vector[0]
  });

  // The result contains the IDs of the similar vectors
  const similarResponses = result.ids;

  return similarResponses;
}

module.exports = { getResponse, getSimilarResponses };
