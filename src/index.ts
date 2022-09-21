import * as Cost from './network/cost';
import * as Activation from './network/activation';
import * as Encoding from './network/encoding';
import * as GradientDescent from './network/gradientDescent';
import * as Network from './network/network';
import * as File from './utils/file';

console.log("Hello, world!");

const MNISTCSVReader = File.readCSVToLabelledDataPoints(
    'D:\TrainingData\MNIST\mnist_train.csv', 
    [1, 784], 
    i => (typeof i === 'number') ? i : (typeof i === 'string') ? parseFloat(i) : (typeof i === 'boolean') ? (i ? 1 : 0) : 0, 
    0
);

const max = 10;
async function *printDataPoints() {
    let i = 0;
    for await (const dataPoint of MNISTCSVReader) {
        console.log(dataPoint);
        yield;
        if (++i >= max) {
            break;
        }
    }
};

printDataPoints();