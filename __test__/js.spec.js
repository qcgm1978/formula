// Just disables the warning, doesn't enable AVX/FMA
// export TF_CPP_MIN_LOG_LEVEL=2
const tf = require('@tensorflow/tfjs-node')
it(`tf`, () => {
  expect(tf instanceof Object).toBeFalsy()
  expect(tf.tidy).toBeInstanceOf(Function)
  expect(tf.scalar).toBeInstanceOf(Function);
  s = tf.scalar(1)
  expect(s.rank).toEqual(0)
  expect(1).toEqual(s.dataSync()[0])
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
    return a;
  }, returnCost, varList);
  const data2 = a.dataSync();
  expect(data2).not.toEqual(data1)
  console.log(data1, data2)
  const f = () => optimizer.minimize(() => {
    // Need to return the loss i.e how bad is our prediction from the correct answer.
  }, returnCost, varList);
  const f1 = () => optimizer.minimize(() => {
    // Need to return the loss i.e how bad is our prediction from the correct answer.
    return tf.scalar(1)
  });
  expect(f).toThrowError('The result y returned by f() must be a tensor')
  expect(f1).toThrowError('Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize().')

});
it(`minimize`, () => {
  const a = tf.variable(tf.scalar(Math.random()));
  const optimizer = tf.train.sgd(.5);
  const f = () => optimizer.minimize(() => {
    // Need to return the loss i.e how bad is our prediction from the correct answer.
    return a.add(tf.tensor1d([1, 2, 3]))
  });
  expect(f).toThrowError('The f passed in variableGrads(f) must return a scalar, but it returned a rank-1 tensor');
})
