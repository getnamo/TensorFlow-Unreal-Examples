var canvas, context, tool;
var socket = io();
var drawing = {};
drawing.strokes = [];

socket.on('connect', function(msg){
	console.log('connected');
});

tool = new tool_pencil();
var drawingTimeout;

function clearCanvas(){
	context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawingComplete()
{
	drawingTimeout = undefined;
	clearCanvas();

	//get min/max
	min = {};
	min.x = 10000;
	min.y = 10000;

	max = {};
	max.x = 0;
	max.y = 0;
	drawing.min = min;
	drawing.max = max;

	for(var i=0; i<drawing.strokes.length; i++){
		for(var j=0; j<drawing.strokes[i].length;j++)
		{
			var point = drawing.strokes[i][j];
			if(point.x < min.x)
			{
				min.x = point.x;
			}
			if(point.y < min.y)
			{
				min.y = point.y;
			}
			if(point.x > max.x)
			{
				max.x = point.x;
			}
			if(point.y > max.y)
			{
				max.y = point.y;
			}
		}
	}

	var border = 40;
	min.x = min.x - border;
	min.y = min.y - border - 10;

	max.x = max.x + border;
	max.y = max.y + border + 10;

	drawing.min = min;
	drawing.max = max;

	result = {};
	result.strokes = [];

	//normalize the drawing to 28/28
	for(var i=0; i<drawing.strokes.length; i++){
		
		var stroke = {}
		stroke.stroke = [];
		result.strokes.push(stroke);

		for(var j=0; j<drawing.strokes[i].length;j++)
		{
			var point = drawing.strokes[i][j];
			var normpt = {};
			var rangex = drawing.max.x - drawing.min.x;
			var rangey = drawing.max.y - drawing.min.y;
			normpt.x = (point.x - drawing.min.x) / rangex * 28.0;
			normpt.y = (point.y - drawing.min.y) / rangey * 28.0;

			result.strokes[i].stroke.push(normpt);
		}
	}

	//var tempresult = {};
	//tempresult.strokes = drawing.strokes;
	//tempresult.min = drawing.min;
	//tempresult.max = drawing.max;

	//console.log(tempresult);
	console.log(result);

	//send data to socket.io
	socket.emit('drawing', result);
	console.log('drawing complete');

	drawing.strokes = [];	//clear strokes	
}

function tool_pencil () {
	var tool = this;
	this.started = false;
	
	canvas = document.getElementById('canvas');
	var d1 = document.getElementById('d1');
	
	canvas.setAttribute('width', $('#canvas').width());
	canvas.setAttribute('height', $('#canvas').height());
	context = canvas.getContext('2d');
	
	this.setXY = function (ev){

		//touch event
		if(ev.offsetX == undefined)
		{
			ev._x = ev.touches.item(0).clientX - $('#d1').offset().left;
			ev._y = ev.touches.item(0).clientY - $('#d1').offset().top;
		}
		else
		{
			ev._x = ev.offsetX;
			ev._y = ev.offsetY;
		}
	};
	
	this.touchstart = function (ev) {
		if(drawingTimeout != undefined){	//restart on draw start
			console.log('drawing timeout reset')
			clearTimeout(drawingTimeout);
		}

		context = canvas.getContext('2d');
		context.beginPath();
		context.moveTo(ev._x, ev._y);
		tool.started = true;

		//add a new stroke
		var firstPoint = {'x': ev._x, 'y': ev._y};
		drawing.strokes.push([firstPoint]);
	};
	this.mousedown = this.touchstart;

	this.touchmove = function (ev) {
	  if (tool.started) {
		context.lineTo(ev._x, ev._y);
		//console.log(ev.offsetX + ', ' + ev.offsetY);
		context.stroke();

		var point = {'x': ev._x, 'y': ev._y};
		drawing.strokes[drawing.strokes.length-1].push(point);

		if(drawingTimeout != undefined){	//restart on draw start
			clearTimeout(drawingTimeout);
		}
		drawingTimeout = setTimeout(drawingComplete, 1000);
	  }
	};
	this.mousemove = this.touchmove;

	// This is called when you release the mouse button.
	this.touchend = function (ev) {
		if (tool.started) {
			tool.touchmove(ev);
			tool.started = false;

			drawingTimeout = setTimeout(drawingComplete, 1000);
	  	}
	};
	this.mouseup = this.touchend;
	this.touchleave = this.touchend;
	this.touchcancel = this.touchend;
}

function evt_lstr(e) {
			
	//Drawing call
	var func = tool[e.type];
	if (func) {
		tool.setXY(e);
		func(e);
	}

	//prevent ios/android resizing with touch gestures
	e.preventDefault();
	
	//log our stuff
	/*if(e.type != "mousemove" && e.type != "touchmove")
	{
	var div = document.createElement("div"), date = new Date();
	div.innerHTML = date.getHours() + ":" + ("0" + date.getMinutes()).slice(-2) + ":"
		  + ("0" + date.getSeconds()).slice(-2) + "." + ("00" + date.getMilliseconds()).slice(-3) + " " + e.type + ', (' + e._x + ', ' + e._y + ') ';

	//Max spam to 10 elements
	d2.appendChild(div);
	if(d2.children.length > 10){
	d2.removeChild(d2.children[0]);
	}
	}
	*/
}


var d1 = document.getElementById("d1"), d2 = document.getElementById("d2");["mouseup","mousedown","mousemove","mouseout","click","dblclick","touchstart","touchend",
		 "touchleave","touchmove","touchcancel"].forEach(function(te) {
	d1.addEventListener(te, evt_lstr);
 });

document.getElementById("clr").addEventListener("click", function(e) {
	d2.innerHTML = "";
	e.preventDefault();
	//clearCanvas();
	drawingComplete();
});