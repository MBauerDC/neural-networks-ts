export type ActivationFunction = (x: number) => number;
export type ActivationFunctionDerivative = (x: number) => number;

export class ActivationFunctionClass {
    constructor(public calculate: ActivationFunction, public derivative: ActivationFunctionDerivative, public name: string) {}
}

export const sigmoidDerivative: ActivationFunctionDerivative = (x: number) => sigmoid.calculate(x) * (1 - sigmoid.calculate(x));
export const sigmoid = new ActivationFunctionClass((x: number) => 1 / (1 + Math.exp(-x)), sigmoidDerivative, "sigmoid");

export const tanhDerivative: ActivationFunctionDerivative = (x: number) => 1 - Math.pow(tanh.calculate(x), 2);
export const tanh = new ActivationFunctionClass((x: number) => Math.tanh(x), tanhDerivative, "tanh");

export const reluDerivative: ActivationFunctionDerivative =  (x: number) => x > 0 ? 1 : 0;
export const relu = new ActivationFunctionClass((x: number) => Math.max(0, x), reluDerivative, "relu");

export const leakyReluDerivative: ActivationFunctionDerivative = (x: number) => x > 0 ? 1 : 0.01;
export const leakyRelu = new ActivationFunctionClass((x: number) => x > 0 ? x : 0.01 * x, leakyReluDerivative, "leakyRelu");

export const linearDerivative: ActivationFunctionDerivative = (x: number) => 1;
export const linear = new ActivationFunctionClass((x: number) => x, linearDerivative, "linear");

export const softmaxDerivative: ActivationFunctionDerivative = (x: number) => 1;
export const softmax = new ActivationFunctionClass((x: number) => Math.exp(x), softmaxDerivative, "softmax");

export const softplusDerivative = (x: number) => 1 / (1 + Math.exp(-x));
export const softplus = new ActivationFunctionClass((x: number) => Math.log(1 + Math.exp(x)), softplusDerivative, "softplus");

export const hardSigmoidDerivative = (x: number) => x > -2.5 && x < 2.5 ? 0.2 : 0;
export const hardSigmoid = new ActivationFunctionClass((x: number) => Math.max(0, Math.min(1, 0.2 * x + 0.5)), hardSigmoidDerivative, "hardSigmoid");

export const eluDerivative = (x: number) => x > 0 ? 1 : elu.calculate(x);
export const elu = new ActivationFunctionClass((x: number) => x > 0 ? x : Math.exp(x) - 1, eluDerivative, "elu");

export const seluDerivative = (x: number) => x > 0 ? 1 : selu.calculate(x) + 1.0507009873554804934193349852946;
export const selu = new ActivationFunctionClass((x: number) => x > 0 ? 1.0507009873554804934193349852946 * x : 1.0507009873554804934193349852946 * (Math.exp(x) - 1), seluDerivative, "selu");

export const softExponentialDerivative = (x: number) => x > 0 ? Math.exp(-x) : Math.exp(x);
export const softExponential = new ActivationFunctionClass((x: number) => x > 0 ? Math.exp(-x) : -Math.exp(x), softExponentialDerivative, "softExponential");

export const softShrinkDerivative = (x: number) => x > 0.5 || x < -0.5 ? 1 : 0;
export const softShrink = new ActivationFunctionClass((x: number) => x > 0.5 ? x - 0.5 : x < -0.5 ? x + 0.5 : 0, softShrinkDerivative, "softShrink");

export const softSignDerivative = (x: number) => 1 / Math.pow(1 + Math.abs(x), 2);
export const softSign = new ActivationFunctionClass((x: number) => x / (1 + Math.abs(x)), softSignDerivative, "softSign");

export const hardTanhDerivative = (x: number) => x > -1 && x < 1 ? 1 : 0;
export const hardTanh = new ActivationFunctionClass((x: number) => Math.max(-1, Math.min(1, x)), hardTanhDerivative, "hardTanh");

export const hardShrinkDerivative = (x: number) => x > 0.5 || x < -0.5 ? 1 : 0;
export const hardShrink = new ActivationFunctionClass((x: number) => x > 0.5 ? x : x < -0.5 ? x : 0, hardShrinkDerivative, "hardShrink");


export class ActivationFunctions {
    static sigmoid = sigmoid;
    static tanh = tanh;
    static relu = relu;
    static leakyRelu = leakyRelu;
    static linear = linear;
    static softmax = softmax;
    static softplus = softplus;
    static hardSigmoid = hardSigmoid;
    static elu = elu;
    static selu = selu;
    static softExponential = softExponential;
    static softShrink = softShrink;
    static softSign = softSign;
    static hardTanh = hardTanh;
    static hardShrink = hardShrink;
}
