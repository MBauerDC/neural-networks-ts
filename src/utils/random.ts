import { Dimension, GenericMutableMatrix, MutableMatrix } from "../matrices/matrix";
import { ActivationFunctionClass } from "../network/activation";

export function boxMullerRandom(): number {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) return boxMullerRandom() // resample between 0 and 1
    return num
  }

export function heInitialization(size: number): number {
    return boxMullerRandom() * Math.sqrt(2 / size);
}

export function normalizedXavierInitialization(inputSize: number, outputSize: number): number {
    const sqrtNPlusM = Math.sqrt(inputSize + outputSize);
    const sqrt6 = Math.sqrt(6);
    const sqrt6DividedBySqrtNPlusM = sqrt6 / sqrtNPlusM;
    const [lower, upper] = [-sqrt6DividedBySqrtNPlusM, sqrt6DividedBySqrtNPlusM];
    const rnd = boxMullerRandom();
    return rnd * (upper - lower) + lower;
}

export function xavierInitialization(inputSize: number) : number {
    const sqrtN = Math.sqrt(inputSize);
    const sqrtNInv = 1 / sqrtN;
    const [lower, upper] = [-sqrtNInv, sqrtNInv];
    const rnd = boxMullerRandom();
    return rnd * (upper - lower) + lower;    
}

export function initializeWeightMatrix<N extends Dimension, M extends Dimension>(n: N, m: M, targetLayerActivationFn: ActivationFunctionClass): MutableMatrix<N,M,number> {
    const matrixData: number[][] = [];
    const size = n * m;
    const activationFunctionName = targetLayerActivationFn.name;
    let initFn = () => xavierInitialization(m);
    if (activationFunctionName.match(/aigmoid|tanh/i)) {
        initFn = () => normalizedXavierInitialization(m, n);
    } else if (activationFunctionName.match(/elu|leakyRelu|relu|selu|softplus|softsign|swish/i)) {
        initFn = () => heInitialization(m);
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            matrixData[i][j] = initFn();
        }
    }
    return new GenericMutableMatrix<N, M, number>(matrixData, null, null, n, m);
}