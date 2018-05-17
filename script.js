// Training data
const data = {
  x: [1, 2, 3, 4],
  y: [1, 3, 5, 7]
}

// Plot for it.
const trace = {
  x: data.x,
  y: data.y,
  mode: 'markers',
  name: 'Training Data',
  marker: { size: 12 }
};

var layout = {
  xaxis: {
    range: [ 0, 6 ]
  },
  yaxis: {
    range: [0, 10]
  }
};

// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d(data.x, [4, 1]);
const ys = tf.tensor2d(data.y, [4, 1]);

model.fit(xs, ys).then(() => predict(5));

function predict(what) {
  if (!what) {
    what = document.getElementById('input').value;
  }
  
  // Use the model to do inference on a data point the model hasn't seen before:
  const prediction = model.predict(tf.tensor2d([what], [1, 1])).asScalar();

  // Plot it.
  const trace2 = {
    x: [what],
    y: [].concat(prediction.get()),
    mode: 'markers',
    name: 'Guess',
    marker: { size: 12 }
  };

  Plotly.newPlot('graph', [trace, trace2], layout);
}

function onPredictClick() {
}