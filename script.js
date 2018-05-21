let MODEL;
init();

function plot(data, isTraining) {
  const trace = {
    x: data.x,
    y: data.y,
    mode: isTraining ? 'lines+markers' : 'markers',
    name: isTraining ? 'Training' : 'Prediction',
    marker: { size: 12, color: isTraining ? '#29B6F6' : '#F06292' }
  };
  
  Plotly.addTraces('graph', trace);
}  
  
function init() {
  Plotly.newPlot('graph', [], {});
  
  // Training data
  const training = {
    x: [1, 2, 3, 4],
    y: [5, 12, 8, 10]  
  }
  plot(training, true);
  
  MODEL = learn(training.x, training.y);  
  //const prediction = guess(model, 5);
  //plot(prediction, false);
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
  model.fit(xs, ys);
  
  //model.fit(xs, ys).then(() => predict(5));
  return model;
}

function predict(what) {
  if (!what) {
    what = document.getElementById('input').value;
  }
  debugger
  
  // Use the model to do inference on a data point the model hasn't seen before:
  const prediction = MODEL.predict(tf.tensor2d([what], [1, 1])).asScalar();
  plot({x: what, y: prediction.get()}, false);
  // predictions.x.push(what);
  // predictions.y.push(prediction.get());
 
}