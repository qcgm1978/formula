const tf = require('@tensorflow/tfjs-node')
// tf = require('tfjs');
it(`tidy`, () => {
  expect(tf instanceof Object).toBeFalsy()
  expect(tf.tidy).toBeInstanceOf(Function)
  expect(tf.scalar).toBeInstanceOf(Function);
  const a = tf.variable(tf.scalar(Math.random()));
  const data1 = a.dataSync()[0];
  const optimizer = tf.train.sgd(.5);
  optimizer.minimize(() => {
    pred = tf.tidy(() => {
      // The value returned inside the tidy function will return
      // through the tidy, in this case to the variable pred.
      const one = tf.scalar(1);
      return a.add(one);
    });
    // Need to return the loss i.e how bad is our prediction from the 
    // correct answer. The optimizer will then adjust the value of a
    // to minimize this loss.
    return pred.sub(1).square().mean();
  });
  const data2 = a.dataSync()[0];
  expect(data2).not.toEqual(data1)
  console.log(data1, data2)
});
