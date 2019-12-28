// import * as cocoSsd from '@tensorflow-models/coco-ssd'
// const tf = require('@tensorflow/tfjs-node')
// import fs from 'fs'
//
// // const dogImage = fs.readFileSync('./dog.mp4.png')
// // const dogImage = fs.readFileSync('./dog.mp4.jpg')
// const dogImage = fs.readFileSync('./dog5.jpg')
//
// const main = async () => {
//   const model = await cocoSsd.load()
//
//   const imageArray = Uint8Array.from(dogImage)
//
//   // const tensor3d = tf.node.decodePng(imageArray);
//   const tensor3d = tf.node.decodeJpeg(imageArray);
//
//   const predictions = await model.detect(tensor3d)
//
//   console.log('Predictions: ')
//   console.log(predictions)
// }
//
// main()

const tf = require('@tensorflow/tfjs-node')
import fs from 'fs'

// const dogImage = fs.readFileSync('./dog.mp4.png')
// const dogImage = fs.readFileSync('./dog.mp4.jpg')
const dogImage = fs.readFileSync('./dog5.jpg')

const main = async () => {
  // const model = await loadGraphModel('./')
  const model = await tf.loadLayersModel('file://path/to/my-model/model.json');

  const imageArray = Uint8Array.from(dogImage)

  // const tensor3d = tf.node.decodePng(imageArray);
  const tensor3d = tf.node.decodeJpeg(imageArray);

  const predictions = await model.detect(tensor3d)

  console.log('Predictions: ')
  console.log(predictions)
}

main()
