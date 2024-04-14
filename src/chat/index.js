const { OpenAI } = require('openai');
require('dotenv').config();


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
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


  return response;
}

module.exports = { getResponse };
