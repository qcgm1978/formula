// Sup globals. Fight me.
let model;
let plotData = {training: {x:[], y:[]}, prediction: {x:[], y:[]}};

init();

function plot() {
  const trace1 = {
    x: plotData.training.x,
    y: plotData.training.y,
    mode: 'lines+markers',
    name: 'Training',
    marker: { size: 12, color:'#29B6F6' }
  };

  const trace2 = {
    x: plotData.prediction.x,
    y: plotData.prediction.y,
    mode: 'markers',
    name: 'Prediction',
    marker: { size: 12, color: '#F06292' }
  };
  Plotly.newPlot('graph', [trace1, trace2], {});
}  
  
function init() {
  // Training data
  const training = generateData(50);
  plotData.training = training;

  learn(training.x, training.y);  
}

function generateData(points) {
  let x = [];
  let y = [];
  
  for (let i = 0; i < points; i++) {
    x[i] = i;
    y[i] = x[i];
  }
  
  return {x:x, y:y}
}

function learn(xData, yData) {
  // Define a model for linear regression.
  model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training.
  const xs = tf.tensor2d(xData, [xData.length, 1]);
  const ys = tf.tensor2d(yData, [yData.length, 1]);
  model.fit(xs, ys).then(plot);
  return model;
}

function predict(what) {
  if (!what) {
    what = document.getElementById('input').value;
  }
  
  // Use the model to do inference on a data point the model hasn't seen before:
  const prediction = model.predict(tf.tensor2d([what], [1, 1])).asScalar();
  console.log(prediction);
  
  plotData.prediction.x.push(what);
  plotData.prediction.y.push(prediction.get());
  console.log(what, prediction.get());
  plot();
}