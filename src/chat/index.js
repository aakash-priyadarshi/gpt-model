const { OpenAI } = require('openai');
require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

async function getResponse(prompt) {
  const config = {
    model: 'ft:davinci-002:liverpool:group-project:9E7LOTDF',
    prompt: prompt,
    max_tokens: 100
  };

  const completion = await openai.completions.create(config);
  let response = '';

  if (completion.choices && completion.choices.length > 0) {
    response = completion.choices[0].text;
  }

  return response;
}

module.exports = { getResponse };
