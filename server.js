const express = require('express');
const http = require('http');
const socketIo = require('socket.io')

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

const { queryAndProcess } = require('./getDataFromFire');

app.use(express.static('public'));

io.on('connection', (socket) => {
    console.log('A user connected');

    socket.on('disconnect', () => {
        console.log('A user disconnected');
    });

    // Get stocks chart's data request from client, get data from server according request and response
    socket.on('chartRequest', (data) => {
        // date to timestamp
        // the timestamp in database is "second", but it's "millisecond" in js, need to convert.
        const startTimestamp = Math.floor(new Date(data.startDate).getTime()/1000)
        const endTimestamp = Math.floor(new Date(data.endDate).getTime()/1000)

        // get data from gcp firebase
        queryAndProcess(startTimestamp, endTimestamp, data.compCode).then(result => {
            console.log(result)
            // return result to client
            socket.emit('chartResponse', result.documents)
        });
    });
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => console.log(`Server is running on port ${PORT}`));
