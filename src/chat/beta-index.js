const express = require('express');
const cors = require('cors');
const { createServer } = require('http');
const { Server } = require('socket.io');
const { OpenAI } = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
const { upload, extractText } = require('./fileHandler'); // Import the file handling module
const { saveFeedback } = require('./feedback');
const bodyParser = require('body-parser');

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

//optional function, can remove this one
app.use((req, res, next) => {
  console.log(`Received ${req.method} request on ${req.url}`); // Logs the method and path of all incoming requests
  next();
});


app.use(cors());
app.use(express.static('public'));
app.use(bodyParser.json());
// Temporary storage for extracted text, keyed by socket ID
let extractedTextStorage = {};

io.on('connection', (socket) => {
  console.log('a user connected');

  // Endpoint to handle file upload directly via sockets
  socket.on('file upload', async (fileData, callback) => {
    const text = await extractText(fileData.path, fileData.mimetype);
    if (text) {
      extractedTextStorage[socket.id] = text; // Store extracted text
      callback({ success: true, message: 'Text extracted successfully', text: text });
    } else {
      callback({ success: false, message: 'Failed to extract text' });
    }
  });

  socket.on('chat message', async (message) => {
    console.log("Received message on server:", message);
    try {
      // Use any stored text for this user as context
      const context = extractedTextStorage[socket.id] || '';
      delete extractedTextStorage[socket.id]; // Clear after use

      const similarResponses = await getSimilarResponses(message);
      if (similarResponses.length > 0) {
        context += '\n' + similarResponses.join('\n');
      }

      const response = await getResponse(message, context);
      socket.emit('chat message', response);
    } catch (error) {
      console.error('Error:', error);
      socket.emit('chat message', 'Error: An error occurred on the server.');
    }
  });

  socket.on('disconnect', () => {
    console.log('user disconnected');
    delete extractedTextStorage[socket.id]; // Ensure clean-up on disconnect
  });
});

//code to handle the upload file
app.post('/upload', upload.single('file'), async (req, res) => {
  console.log('/upload route hit');
  if (!req.file) {
      console.log('No file part in the request.');
      return res.status(400).send('No file uploaded.');
  }
  console.log(`File uploaded: ${req.file.filename}`); // Log the filename of the uploaded file

  try {
      const extractedText = await extractText(req.file.path, req.file.mimetype);
      if (extractedText) {
          console.log('Text extraction successful');
          io.to(req.body.socketId).emit('file text', extractedText); // Emit to specific socket ID
          res.send({ message: 'File processed successfully.', text: extractedText });
      } else {
          console.log('Failed to extract text from file');
          res.status(500).send('Failed to process the file.');
      }
  } catch (error) {
      console.error('Error processing file:', error);
      res.status(500).send('Server error processing file');
  }
});


// code to handle feedback
app.post('/feedback', async (req, res) => {
  const { session_id, question, response, feedback } = req.body;

  if (typeof session_id !== 'string' || typeof question !== 'string' || typeof response !== 'string' || typeof feedback !== 'boolean') {
    return res.status(400).send('Invalid request data');
  }

  try {
    const result = await saveFeedback({ session_id, question, response, feedback });

    if (result.success) {
      res.status(201).send(result.message);
    } else {
      res.status(500).send(result.message);
    }
  } catch (error) {
    console.error('Server error when processing feedback:', error);
    res.status(500).send('Server error processing feedback');
  }
});


// function to get response from the model
async function getResponse(message, context = '') {  // Accept context as a parameter
  const config = {
    model: 'ft:gpt-3.5-turbo-0125:personal:group-project:9GVOiDii',
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

// feature for topic search

// Function to classify text into topics
async function classifyText(text, topics) {
  const classifier = await import('@xenova/transformers');
  const nlp = await classifier.pipeline('zero-shot-classification', {
    model: 'facebook/bart-large-mnli'
  });
  return nlp({
    inputs: text,
    candidate_labels: topics, // Predefined or dynamic list of topics
    multi_class: true
  });
}

socket.on('chat message', async (message) => {
  console.log("Received message on server:", message);
  try {
    // Define possible topics
    const topics = ["technology", "health", "finance", "education", "entertainment"];

    // Classify the message into topics
    const topicResults = await classifyText(message, topics);
    console.log("Classified topics:", topicResults);

    // Use any stored text for this user as context
    const context = extractedTextStorage[socket.id] || '';
    delete extractedTextStorage[socket.id]; // Clear after use

    const similarResponses = await getSimilarResponses(message);
    if (similarResponses.length > 0) {
      context += '\n' + similarResponses.join('\n');
    }

    const response = await getResponse(message, context);
    // Include the topics in the response to the client
    socket.emit('chat message', { text: response, topics: topicResults.labels });
  } catch (error) {
    console.error('Error:', error);
    socket.emit('chat message', { text: 'Error: An error occurred on the server.', topics: [] });
  }
});


//

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