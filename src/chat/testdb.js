const mongoose = require('mongoose');
const feedbackSchema = new mongoose.Schema({
  // Define your schema as needed
  text: String,
  date: { type: Date, default: Date.now }
});
const Feedback = mongoose.model('Feedback', feedbackSchema);

// Replace 'your_connection_string' with your actual connection string
mongoose.connect('mongodb+srv://aakashm301:SvbbMtmMJAkidsP4@cluster0.tzpt4jf.mongodb.net/chat', { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch(err => console.error('Could not connect to MongoDB', err));

const feedback = new Feedback({ text: 'This is a test feedback entry' });

feedback.save()
  .then(doc => console.log('Document inserted', doc))
  .catch(err => console.error('Error inserting document', err));
