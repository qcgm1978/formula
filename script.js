// Training data
const training = {
  x: [1, 2, 3, 4],
  y: [5, 12, 8, 10]
}

const predictions = { x: [], y: []};

// Plot for it.
const trace = {
  x: training.x,
  y: training.y,
  mode: 'lines+markers',
  name: 'Training Data',
  marker: { size: 12 }
};

var layout = {
};

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d(training.x, [4, 1]);
const ys = tf.tensor2d(training.y, [4, 1]);

model.fit(xs, ys).then(() => predict(5));

function predict(what) {
  if (!what) {
    what = document.getElementById('input').value;
  }
  
  // Use the model to do inference on a data point the model hasn't seen before:
  const prediction = model.predict(tf.tensor2d([what], [1, 1])).asScalar();
  
  predictions.x.push(what);
  predictions.y.push(prediction.get());
  
  // Plot it.
  const trace2 = {
    x: predictions.x,
    y: predictions.y,
    mode: 'markers',
    name: 'Guess',
    marker: { size: 12 }
  };

  Plotly.newPlot('graph', [trace, trace2], layout);
}