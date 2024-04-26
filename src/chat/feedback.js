const mongoose = require('mongoose');

require('../database/index');
// Define the schema for feedback
const feedbackSchema = new mongoose.Schema({
  session_id: String,
  question: String,
  response: String,
  feedback: Boolean,
}, { collection: 'feedback' }); // Specify the collection name here

// Create the model from the schema
const Feedback = mongoose.model('feedback', feedbackSchema, 'feedback'); // Or specify the collection name here

// Function to save feedback to the database
async function saveFeedback({ session_id, question, response, feedback }) {
  try {
    const newFeedback = new Feedback({ session_id, question, response, feedback });
    await newFeedback.save();
    console.log('Feedback saved successfully');
    return { success: true, message: 'Feedback saved successfully' };
  } catch (error) {
    console.error('Failed to save feedback:', error);
    return { success: false, message: 'Failed to save feedback' };
  }
}

module.exports = { saveFeedback };
