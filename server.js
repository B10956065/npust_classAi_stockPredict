const express = require('express');
const http = require('http');
const socketIo = require('socket.io')

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

app.use(express.static('public'));

io.on('connection', (socket) => {
    console.log('A user connected');

    socket.on('disconnect', () => {
        console.log('A user disconnected');
    });

    // Get stocks chart's data request from client, get data from server according request and response
    socket.on('chartRequest', (data) => {
        const result = `chart for '${data.compCode}' in ${data.startDate} ~ ${data.endDate}`

        // get data from gcp firebase


        // return result to client
        socket.emit('chartResponse', result)
    });


});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => console.log(`Server is running on port ${PORT}`));
