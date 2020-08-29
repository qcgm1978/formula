// Just disables the warning, doesn't enable AVX/FMA
// export TF_CPP_MIN_LOG_LEVEL=2
const tf = require('@tensorflow/tfjs-node')
it(`tf`, () => {
  expect(tf instanceof Object).toBeFalsy()
  expect(tf.tidy).toBeInstanceOf(Function)
  expect(tf.scalar).toBeInstanceOf(Function);
})
it(`The optimizer will then adjust the value of a to minimize this loss`, () => {
  const a = tf.variable(tf.scalar(Math.random()));
  const data1 = a.dataSync();
  const optimizer = tf.train.sgd(.5);
  // Executes f() and minimizes the scalar output of f() by computing gradients of y with respect to the list of trainable variables provided by varList. If no list is provided, it defaults to all trainable variables.
  const returnCost = true
  const varList = [a]
  optimizer.minimize(() => {
    // Need to return the loss i.e how bad is our prediction from the correct answer.
    // return pred.sub(1).square().mean();
    return a;
  }, returnCost, varList);
  const data2 = a.dataSync();
  expect(data2).not.toEqual(data1)
  console.log(data1, data2)
});
