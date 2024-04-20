const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const { OpenAI } = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const index = pc.index('group-project-ai'); //index name for vector database.

let counter = 0;

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer);

app.use(express.static('public'));

io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('chat message', async (message) => {
    try {
      // Search for similar past messages or topics
      const similarResponses = await getSimilarResponses(message);
      let context = '';
  
      if (similarResponses.length > 0) {
        // If similar responses are found, use them as context for the GPT model
        context = similarResponses.join('\n');
      }
      
      const response = await getResponse(message, context);  // Pass the context to the GPT model
      socket.emit('chat message', response);
    } catch (error) {
      console.error('Error:', error);
      socket.emit('chat message', 'Error: An error occurred on the server.');
    }
  });
  

  socket.on('disconnect', () => {
    console.log('user disconnected');
  });
});

// function to get response from the model
async function getResponse(message, context = '') {  // Accept context as a parameter
  const config = {
    model: 'gpt-3.5-turbo',
    stream: true,
    messages: [
      {
        role: 'system',
        content: context  // Use the context in the chat completion request
      },
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
        try {
            await index.namespace('ns1').upsert([{
                id: `item-${counter}`,  // Use the counter as the ID
                values: vectorArray  // Ensure this is a plain array
            }]);
            console.log("Upsert successful");
            counter++;  // Increment the counter
        } catch (error) {
            console.error("Error during upsert:", error);
        }
    } else {
        console.error("Vector Tensor is empty or invalid");
    }
  } catch (error) {
      console.error("Error during vectorization:", error.message);  // Capture any errors during the model operation
  }
  return response;  
}


async function getSimilarResponses(query) {
  // Convert the query to a vector
  const transformers = await import('@xenova/transformers');
  const model = await transformers.pipeline('feature-extraction', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  const vectorTensor = await model(query);

  // Ensure that the vector is a simple array of floats
  const vectorData = Array.from(vectorTensor.data);

  // Check if vectorData is properly formatted and not empty
  if (!vectorData || vectorData.length === 0) {
    console.error("Failed to generate a valid vector from the input.");
    return [];
  }

  try {
    // Query Pinecone for similar vectors
    const result = await index.namespace('ns1').query({
      vector: vectorData,
      topK: 5
    });

    // Check if the result is valid and contains IDs
    if (result && result.ids && result.ids.length > 0) {
      return result.ids;  // Return the IDs of similar vectors
    } else {
      console.error("No similar vectors found or error in querying.");
      return [];
    }
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    return [];
  }
}


module.exports = { getResponse, getSimilarResponses };

const shutDown = () => {
  console.log('Received kill signal, shutting down gracefully');
  httpServer.close(() => {
    console.log('Closed out remaining connections');
    process.exit(0);
  });

  // if after 10 seconds not all connections are closed, shut down forcefully
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', shutDown);
process.on('SIGINT', shutDown);

httpServer.listen(3001, () => {
  console.log('listening on *:3001');
});