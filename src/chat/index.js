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
const index = pc.index('group-project-ai');

let counter = 0;

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer);

app.use(express.static('public'));

io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('chat message', async (message) => {
    try {
      const response = await getResponse(message);
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