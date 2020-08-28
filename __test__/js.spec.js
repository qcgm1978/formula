const tf = require('@tensorflow/tfjs')
// tf = require('tfjs');
it(`tidy`, () => {
  expect(tf instanceof Object).toBeFalsy()
  expect(tf.tidy).toBeInstanceOf(Function)
  expect(tf.scalar).toBeInstanceOf(Function);
});
