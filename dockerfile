# Use the official Node.js 16 image.
# https://hub.docker.com/_/node
FROM node:16

# Create app directory (where your app will live within the container)
WORKDIR /usr/src/app

# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
COPY package*.json ./

RUN npm install
# If you are building your code for production
# RUN npm ci --only=production

# Bundle app source inside Docker image
COPY . .

# Your app binds to port 3001 so you use the EXPOSE instruction to have it mapped by the docker daemon
EXPOSE 3001

# Define the command to run your app using CMD which defines your runtime
CMD [ "node", "src/chat/index.js" ]
