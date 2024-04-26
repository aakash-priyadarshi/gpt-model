// This file would contain your database setup and connection logic.
// For example, if you were using MongoDB with Mongoose, it might look something like this:

const mongoose = require('mongoose');

// Add the database name at the end of the URI path
mongoose.connect('mongodb+srv://aakashm301:SvbbMtmMJAkidsP4@cluster0.tzpt4jf.mongodb.net/chat', {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
.then(() => {
  console.log('Database connected successfully');
  mongoose.set('debug', true); // Enable debugging
})
.catch(err => console.error('Database connection error:', err));

module.exports = mongoose;
