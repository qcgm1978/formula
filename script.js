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
  const training = generateData(10);
  plotData.training = training;
  
  doALearning();
  
}

function generateData(points) {
  // True coefficients
  const a = 0;
  const b = 3;
  const c = 10; 
  const d = -4;
  
  let x = [];
  let y = [];
  
  for (let i = 0; i < points; i++) {
    x[i] = i;
    y[i] = a * i*i*i + b * i*i + c * i + d;
  }
  
  return {x:x, y:y}
}

function doALearning() {
  
  // Based on https://github.com/tensorflow/tfjs-examples/blob/master/polynomial-regression-core/index.js
  
  // We generated some data according to a formula that's up to cubic, so we want
  // to learn the coefficients for
  // y = a * x ^ 3 + b * x^2 + c * x + d
  
  // Step 1. Set up the variables we're trying to learn.
  const a = tf.variable(tf.scalar(Math.random()));
  const b = tf.variable(tf.scalar(Math.random()));
  const c = tf.variable(tf.scalar(Math.random()));
  const d = tf.variable(tf.scalar(Math.random()));
  
  // Step 2. Create an optimizer, we will use this later. You can play
  // with some of these values to see how the model perfoms.
  const numIterations = 75;
  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate); 
  
  // Step 3. Write our training process functions.
  
  // This predicts a value
  function predict(x) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
    return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3, 'int32')))
        .add(b.mul(x.square()))
        .add(c.mul(x))
        .add(d);
    });
  }
  
  // This tells you how good the prediction is based on what you expected.
  function loss(prediction, labels) {
    // Having a good error function is key for training a machine learning model
    const error = prediction.sub(labels).square().mean();
    return error;
  }
  
  // This trains the model.
  async function train(xs, ys, numIterations) {
    for (let iter = 0; iter < numIterations; iter++) {
      // optimizer.minimize is where the training happens.

      // The function it takes must return a numerical estimate (i.e. loss)
      // of how well we are doing using the current state of
      // the variables we created at the start.

      // This optimizer does the 'backward' step of our training process
      // updating variables defined previously in order to minimize the
      // loss.
      optimizer.minimize(() => {
        // Feed the examples into the model
        const pred = predict(xs);
        return loss(pred, ys);
      });

      // Use tf.nextFrame to not block the browser.
      await tf.nextFrame();
    }
  }
  
  
  
}



// function learn(xData, yData) {
//   // Define a model for linear regression.
//   model = tf.sequential();
//   model.add(tf.layers.dense({units: 1, inputShape: [1]}));

//   // Prepare the model for training: Specify the loss and the optimizer.
//   model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

//   // Generate some synthetic data for training.
//   const xs = tf.tensor2d(xData, [xData.length, 1]);
//   const ys = tf.tensor2d(yData, [yData.length, 1]);
//   model.fit(xs, ys).then(plot);
//   return model;
// }

// function predict(what) {
//   if (!what) {
//     what = document.getElementById('input').value;
//   }
  
//   // Use the model to do inference on a data point the model hasn't seen before:
//   const prediction = model.predict(tf.tensor2d([what], [1, 1]));
//   console.log(prediction);
  
//   plotData.prediction.x.push(what);
//   plotData.prediction.y.push(prediction.get());
//   console.log(what, prediction.get());
//   plot();
// }