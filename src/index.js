const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const Chat = require('./chat');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

// Serve static files from the "public" directory
app.use(express.static('public'));

io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('chat message', async (msg) => {
    console.log('message: ' + msg);
    const response = await Chat.getResponse(msg);
    io.emit('chat message', response);
  });

  socket.on('disconnect', () => {
    console.log('user disconnected');
  });
});

server.listen(3000, () => {
  console.log('listening on *:3000');
});
