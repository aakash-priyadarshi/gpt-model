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
  console.log("Response from GPT:", response);

  // Convert the response to a vector
  const transformers = await import('@xenova/transformers');
  const model = await transformers.pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  
  try {
    const vectorTensor = await model(response);
    console.log("Tensor object:", vectorTensor);  // Check the tensor output

    if (vectorTensor && vectorTensor.data && vectorTensor.data.length > 0) {
        const vectorArray = Array.from(vectorTensor.data);
        console.log("Array to be upserted:", vectorArray);  // Further inspect the array format

        // Upsert the vector to Pinecone
        await index.namespace('ns1').upsert([{
            id: `item-${counter}`,  // Use the counter as the ID
            values: vectorArray  // Ensure this is a plain array
        }]);

        counter++;  // Increment the counter
    } else {
        console.error("Vector Tensor is empty or invalid");
    }
} catch (error) {
    console.error("Error during vectorization:", error.message);  // Capture any errors during the model operation
}  
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
