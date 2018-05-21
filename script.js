go();

function plot(training, prediction) {
  const trace1 = {
    x: training.x,
    y: training.y,
    mode: 'lines+markers',
    name: 'Training Data',
    marker: { size: 12, color: '#29B6F6' }
  };
  
   
  // Plot it.
  const trace2 = {
    x: predictions.x,
    y: predictions.y,
    mode: 'markers',
    name: 'Guess',
    marker: { size: 12, color: '#F06292' }
  };

  Plotly.newPlot('graph', [trace1, trace2], layout);

}

const predictions = { x: [], y: []};

// Plot for it.
const trace = {
  x: training.x,
  y: training.y,
  mode: 'lines+markers',
  name: 'Training Data',
  marker: { size: 12, color: '#29B6F6' }
};

var layout = {};
}
  
  
function go() {
  // Training data
  const training = {
    x: [1, 2, 3, 4],
    y: [5, 12, 8, 10]  
  }
  plot(training);
  
  const model = learn(training.x, training.y);  
  predict(model);
}


function learn(xData, yData) {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training.
  const xs = tf.tensor2d(xData, [4, 1]);
  const ys = tf.tensor2d(yData, [4, 1]);

  model.fit(xs, ys).then(() => predict(5));
  return model;
}

function predict(model, what) {
  if (!what) {
    what = document.getElementById('input').value;
  }
  
  // Use the model to do inference on a data point the model hasn't seen before:
  const prediction = model.predict(tf.tensor2d([what], [1, 1])).asScalar();
  
  predictions.x.push(what);
  predictions.y.push(prediction.get());
 
}