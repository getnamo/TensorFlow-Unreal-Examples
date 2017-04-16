/** TensorFlow-ue4 example server for hosting both webclients and UE4 in a connected architecture. 
	Images sent from the web will be forwarded to the last connected UE4 client for decoding. 
	See webclient/index.js for details on how the drawing website is done.*/

//External modules
const app = require('express')();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const util = require('util');
var clients = [];


//Send all the webclient resources
app.get('/', function(req, res){
  res.sendFile(__dirname + '/webclient/index.html');
});
app.get('/webclient/index.js', function(req, res){
  res.sendFile(__dirname + '/webclient/index.js');
});
app.get('/webclient/styles.css', function(req, res){
  res.sendFile(__dirname + '/webclient/styles.css');
});

let ue4client = undefined;

//socket.io client and event handling
io.on('connection', function(socket){
	//track connected clients via log
	clients.push(socket.id);
	var clientConnectedMsg = 'User connected ' + util.inspect(socket.id) + ', total: ' + clients.length;
	var clientDisconnectedMsg = 'User disconnected ' + util.inspect(socket.id) + ', total: ' + clients.length;

	console.log(clientConnectedMsg);

	//track disconnected clients via log
	socket.on('disconnect', function(){

		//Pop the client
		clients.pop(socket.id);

		//did ue4 quit? clear it
		if(socket == ue4client)
		{
			ue4client = undefined;
		}

		console.log(clientDisconnectedMsg);
	});

	//notify when connected that this is the ue4 client
	socket.on('ue4client', function(){
		console.log('ue4 connected');
		ue4client = socket;
	});

	socket.on('drawing', function(drawing)
	{
		//send our strokes over
		console.log('received drawing with ' + drawing.strokes.length + ' strokes.');
		if(ue4client){
			ue4client.emit('drawing', drawing);
			console.log('emitted drawing to ue4');
		}
	});

});

//make the server listen on port 3000
http.listen(3000, function(){
  console.log('listening on *:3000');
});
