import { Column, Dimension, Matrix, GenericMatrix, multiplyMatrices } from "../../node_modules/matrices/src/matrix";
import { MutableMatrix, MutableColumn, GenericMutableMatrix, GenericMutableColumn, GenericMutableRow } from "../../node_modules/matrices/src/mutable";
import { toMutable } from "../../node_modules/matrices/src/sparseMutable";
import { initializeWeightMatrix } from "../utils/random";
import { ActivationFunctionClass, ActivationFunctions } from "./activation";
import { LearningError } from "./cost";

type DataPoint<N extends Dimension> = { input: Column<N, number> }
type LabelledDataPoint<N extends Dimension, M extends Dimension> = { input: Column<N, number>, output: Column<M, number> }
type DataPointGenerator<N extends Dimension> = Generator<DataPoint<N>>;
type LabelledDataPointGenerator<N extends Dimension, M extends Dimension> = Generator<LabelledDataPoint<N, M>>;

type DataSet<N extends Dimension, M extends Dimension> = Set<DataPoint<N>> | Set<LabelledDataPoint<N, M>>;
type LabelledDataSet<N extends Dimension, M extends Dimension> = Set<LabelledDataPoint<N, M>>;
type DataSequence<N extends Dimension, M extends Dimension> = Array<DataPoint<N> | LabelledDataPoint<N, M>>;
type LabelledDataSequence<N extends Dimension, M extends Dimension> = Array<LabelledDataPoint<N, M>>;

function labelledDataPointGeneratorFromSequence<N extends Dimension, M extends Dimension>(sequence: LabelledDataSequence<N, M>): LabelledDataPointGenerator<N, M> {
    return (function* () {
        for (const dataPoint of sequence) {
            yield dataPoint;
        }
    })();
}

function labelledDataPointGeneratorFromSet<N extends Dimension, M extends Dimension>(set: LabelledDataSet<N, M>): LabelledDataPointGenerator<N, M> {
    return (function* () {
        for (const dataPoint of set) {
            yield dataPoint;
        }
    })();
}

type ConnectionMatrix<O extends Dimension, N extends Dimension, F extends number> = MutableMatrix<O, N, F>;


type Layer<N extends Dimension, F extends number> = {
    n: N,
    activations: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>;
    biases: MutableColumn<N, F>;
    activationFunction: ActivationFunctionClass;
    setActivations(values: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>): void;
    setBiases(biases: MutableColumn<N, F>): void;
    getPreviousLayer(): Layer<Dimension, F> | undefined;
    getPreviousLayerConnection(): ConnectionMatrix<N, Dimension, F> | undefined
    getNextLayer(): Layer<Dimension, F> | undefined;
    getNextLayerConnection(): ConnectionMatrix<Dimension, N, F> | undefined
    setPreviousLayer<I extends Dimension>(layer: Layer<I, F>, weights: ConnectionMatrix<N, I, F>): void;
    setNextLayer<O extends Dimension>(layer: Layer<O, F>, weights: ConnectionMatrix<O, N, F>) : void;
    feedToNextLayer(): void;
    feedForward(): void;
    getSummedInputs(): MutableColumn<N, F> | MutableMatrix<N, Dimension, F> | undefined;
    setSummedInputs(summedInputs: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>): void
}

class GenericLayer<N extends Dimension, F extends number> implements Layer<N, F> {
    private _previousLayer: Layer<Dimension, F> | undefined;
    private _nextLayer: Layer<Dimension, F> | undefined;
    private _previousLayerConnection: ConnectionMatrix<N, Dimension, F> | undefined;
    private _nextLayerConnection: ConnectionMatrix<Dimension, N, F> | undefined;
    private _summedInputs: MutableColumn<N, F> | MutableMatrix<N, Dimension, F> | undefined;
    public readonly n: N;

    constructor(
        public activations: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>,
        public biases: MutableColumn<N, F>,
        public readonly activationFunction: ActivationFunctionClass
    ) {
        this.n = activations.n;
    }

    getPreviousLayer(): Layer<number, F> | undefined {
        return this._previousLayer;
    }

    getPreviousLayerConnection(): ConnectionMatrix<N, number, F> | undefined {
        return this._previousLayerConnection;
    }

    getNextLayer(): Layer<number, F> | undefined {
        return this._nextLayer;
    }

    getNextLayerConnection(): ConnectionMatrix<number, N, F> | undefined {
        return this._nextLayerConnection;
    }

    getSummedInputs(): MutableColumn<N, F> | MutableMatrix<N, number, F> | undefined {
        return this._summedInputs;
    }

    reset(): void {
        this.activations.scale(0 as F);
        this.biases.scale(0 as F);
        this._summedInputs?.scale(0 as F);
    }

    setActivations(values: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>): void {
        this.activations = values;
    }

    setBiases(biases: MutableColumn<N, F>): void {
        this.biases = biases;
    }

    previousLayer(): Layer<Dimension, F> | undefined { 
        return this._previousLayer; 
    }

    previousLayerConnection(): ConnectionMatrix<N, Dimension, F> | undefined {
        return this._previousLayerConnection;
    }

    nextLayer(): Layer<Dimension, F> | undefined {
        return this._nextLayer;
    }

    nextLayerConnection(): ConnectionMatrix<Dimension, N, F> | undefined {
        return this._nextLayerConnection;
    }

    setPreviousLayer<I extends Dimension>(layer: Layer<I, F>, weights: ConnectionMatrix<N, I, F>): void {
        this._previousLayer = layer;
        this._previousLayerConnection = weights;
    }

    setNextLayer<O extends Dimension>(layer: Layer<O, F>, weights: ConnectionMatrix<O, N, F>): void {
        this._nextLayer = layer;
        this._nextLayerConnection = weights;
    }

    summedInputs(): MutableColumn<N, F> | MutableMatrix<N, Dimension, F> | undefined {
        return this._summedInputs;
    }

    setSummedInputs(summedInputs: MutableColumn<N, F> | MutableMatrix<N, Dimension, F>): void {
        this._summedInputs = summedInputs;
    }

    feedToNextLayer<O extends Dimension>(): void {
        if (this._nextLayer === undefined) {
            return;
        }
        // multiply weight-matrix with values of current layer
        console.log("Current layer values: ", this.activations);
        console.log("Next layer weights: ", this._nextLayerConnection);
        let newValues = toMutable(multiplyMatrices(this._nextLayerConnection as unknown as Matrix<O, N, F>, this.activations as Matrix<N, number, F>)) as MutableMatrix<O, number, F>;
        console.log("New values after multiply: ", newValues);
        // add biases (for each column representing a data point)
        const biasMatrixData = (new Array<MutableColumn<Dimension, F>>(newValues.m)).fill(this._nextLayer.biases);
        const biasMatrix = new GenericMatrix<O, Dimension, F>(null, null, biasMatrixData as unknown as Column<O, F>[], this._nextLayer.biases.n as O, newValues.m);
        newValues.add(biasMatrix);
        this._nextLayer.setSummedInputs(newValues)
        console.log("New values after add biases: ", newValues);
        // apply activation function
        const newActivations = newValues.mapped((f: F) => this._nextLayer?.activationFunction.calculate(f) as F);
        console.log("New values after activation Function: ", newActivations);
        // set new values
        this._nextLayer.setActivations(newActivations);
    }

    feedForward(): void {
        if (this._nextLayer === undefined) {
            return;
        }
        this.feedToNextLayer();
        this._nextLayer.feedForward();
    }
}

const initialLayer = new GenericLayer<2, number>(new GenericMutableColumn([0, 0], 2), new GenericMutableColumn([0, 0], 2), ActivationFunctions.relu);
const hiddenLayer = new GenericLayer<3, number>(new GenericMutableColumn([0.2, 0.45, 0.35], 3), new GenericMutableColumn([0, 0, 0], 3), ActivationFunctions.relu);
const outputLayer = new GenericLayer<1, number>(new GenericMutableColumn([0.2], 1), new GenericMutableColumn([0], 1), ActivationFunctions.sigmoid);
const weights1 = new GenericMutableMatrix([[0.2, 0.45], [0.35, 0.2], [0.45, 0.35]], null, null, 3, 2);
const weights2 = new GenericMutableRow([0.2, 0.45, 0.35], 3);
hiddenLayer.setNextLayer(outputLayer, weights2);
initialLayer.setNextLayer(hiddenLayer, weights1);
initialLayer.setActivations(new GenericMutableMatrix([[0.3], [0.9]], null, null, 2, 1));
initialLayer.feedForward();
const output = outputLayer.activations;
console.log(output);

export class Network<I extends Dimension, O extends Dimension> {
    readonly layers: Layer<Dimension, number>[] = [];
    readonly weights: ConnectionMatrix<Dimension, Dimension, number>[] = [];
    readonly i: I;
    readonly o: O;
    constructor(initialLayer: Layer<I, number>) {
        this.weights = [];
        let currLayer: Layer<Dimension, number> = initialLayer;
        this.i = currLayer.n as I;
        let nextLayer: Layer<Dimension, number> | undefined = currLayer.getNextLayer();
        this.layers.push(currLayer);
        let i = 1;
        while (nextLayer !== undefined) {
            this.layers[i] = nextLayer;
            this.weights[1 + 1] = currLayer.getNextLayerConnection()!;
            currLayer = nextLayer;
            nextLayer = currLayer.getNextLayer();
            i++;
        }
        this.o = currLayer.n as O;
    }

    feedThrough(input: Column<I, number> | Matrix<I, Dimension, number>): Column<O, number> | Matrix<O, Dimension, number> {
        this.layers[0].setActivations(toMutable(input) as MutableColumn<I, number>);
        this.layers[0].feedForward();
        return this.layers[this.layers.length - 1].activations as unknown as Column<O, number>;
    }

    getActivations(layerIdx: number): Column<O, number> | Matrix<O, Dimension, number>{
        return this.layers[layerIdx].activations as unknown as Column<O, number> | Matrix<O, Dimension, number>;
    }
    
}



class LayerConnectionLearningData<N extends Dimension> {
    constructor(
      public readonly chainRuleResultByNode: MutableMatrix<N, 1, number>,
      public readonly summedInputValueByNode: MutableMatrix<N, 1, number>,
      public readonly incomingWeightCostGradient: MutableMatrix<N, 1, number>,
      public readonly biasGradientByNode: MutableMatrix<N, 1, number>,
      public readonly weightVelocities: MutableMatrix<N, 1, number>,
      public readonly biasVelocities: MutableMatrix<N, 1, number>
    ){}
}


interface Learner<I extends Dimension, O extends Dimension> {
    network: Network<I, O>;
    train<N extends Dimension, M extends Dimension>(dataPointGenerator: LabelledDataPointGenerator<N, M>, learningRate: number, momentum: number, regularization: number, batchSize: number, epochs: number): void;
    test<N extends Dimension, M extends Dimension>(dataPointGenerator: LabelledDataPointGenerator<N, M>): ClassificationTestReport;
}

class BackpropagationLearner implements Learner {
    protected readonly layerConnectionLearningData: LayerConnectionLearningData<Dimension>[];

    

    protected randomizeWeights(outputLayerIdx: number): void {
        if (outputLayerIdx < 0 || outputLayerIdx >= this.network.layers.length) {
            throw new Error('Invalid layer idx.');
        }
        const outputLayer = this.network.layers[outputLayerIdx];
        const currWeightsToLayer = this.network.weights[outputLayerIdx];
        const activationFn = outputLayer.activationFunction();
        const randomizedWeights = initializeWeightMatrix(currWeightsToLayer.n, currWeightsToLayer.m, activationFn)
        this.network.weights[outputLayerIdx] = randomizedWeights;
    }

    constructor(public readonly network: Network, protected readonly learningError: LearningError<Dimension>) {
        this.layerConnectionLearningData = [];
        for (let i = 1; i < network.layers.length; i++) {
            const connection = network.weights[i];
            this.layerConnectionLearningData[i] = new LayerConnectionLearningData(
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m),
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m),
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m),
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m),
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m),
                new GenericMutableColumn(Array(connection.m).fill(0), connection.m)
            );
        }
    }

    protected makeForwardPass<N extends Dimension, M extends Dimension>(dataPoint: LabelledDataPoint<N, M>): void {
        const inputLayer = this.network.layers[0];
        inputLayer.setValues(dataPoint.input);
        inputLayer.feedForward();
        for (let l = 1; l <= this.network.layers.length; l++) {
            const layer = this.network.layers[l];
            const connection = this.network.weights[l];
            const learningData = this.layerConnectionLearningData[l];
            for (let i = 0; i < connection.m; i++) {
                learningData.summedInputValueByNode.setValue(i, 0, layer.summedInputs().getValue(i, 0)); //optimize by making mutable and replacing
            }
        }
    }

    protected getOutputLayerDistance<N extends Dimension, M extends Dimension>(dataPoint: LabelledDataPoint<N, M>): Column<Dimension, number> {
        const outputLayer = this.network.layers[this.network.layers.length - 1];
        const output = outputLayer.values() as MutableColumn<Dimension, number>;
        return this.learningError.distance(output, dataPoint.output);        
    }

    protected getTotalCost<N extends Dimension>(distance: Column<N, number>): number {
        return this.learningError.distanceCost(distance);
    }

    protected setOutputLayerChainRuleResult<N extends Dimension, M extends Dimension>(dataPoint: LabelledDataPoint<N, M>): void {
        const outputLayer = this.network.layers[this.network.layers.length - 1];
        const learningData = this.layerConnectionLearningData[this.network.layers.length - 1];
        const nodeCostDerivative = this.learningError.nodeCostFunctionDerivative;
        for (let i = 0; i < outputLayer.n; i++) {
            const costDerivative = nodeCostDerivative(outputLayer.values().getValue(i, 0), dataPoint.output.getValue(i, 0));
            const activationDerivative = outputLayer.activationFunction().derivative(learningData.summedInputValueByNode.getValue(i, 0));
            learningData.chainRuleResultByNode.setValue(
                i, 
                0, 
                activationDerivative * costDerivative
            );
        }
    }

    protected setHiddenLayerChainRuleResult<N extends Dimension, M extends Dimension>(layerIndex: number): void {
        if (layerIndex === 0) {
            throw new Error("Cannot set chain rule result for input layer");
        }
        const layer = this.network.layers[layerIndex];
        const layerLearningData = this.layerConnectionLearningData[layerIndex];
        const nextLayer = this.network.layers[layerIndex + 1];
        const nextLayerLearningData = this.layerConnectionLearningData[layerIndex + 1];
        const outgoingWeights = layer.nextLayerConnection();
        const noOfNodesInThisLayer = layer.n;
        const noOfNodesInNextLayer = nextLayer.n;

        for (let thisLayerNodeIndex = 0; thisLayerNodeIndex < noOfNodesInThisLayer; thisLayerNodeIndex++) {
            let newChainRuleResult: number = 0;
            // Sum the chain rule results for each connected next layer node
            for (let nextLayerNodeIndex = 0; nextLayerNodeIndex < noOfNodesInNextLayer; nextLayerNodeIndex++) {
                const weightedInputDerivative = outgoingWeights.getValue(nextLayerNodeIndex, thisLayerNodeIndex);
                const nextLayerNodeChainRuleResult = nextLayerLearningData.chainRuleResultByNode.getValue(nextLayerNodeIndex, 0);
                newChainRuleResult += weightedInputDerivative * nextLayerNodeChainRuleResult;
            }
            // Multiply by the derivative of the activation function evaluated at the current summed input value of the current node
            const currNodeSummedInputValue = layerLearningData.summedInputValueByNode.getValue(thisLayerNodeIndex, 0);
            newChainRuleResult *= layer.activationFunction().derivative(currNodeSummedInputValue);
            layerLearningData.chainRuleResultByNode.setValue(thisLayerNodeIndex, 0, newChainRuleResult);
        }
    }

    protected updateGradients(layerIdx: number): void {
        if (layerIdx === 0) {
            throw new Error("Cannot update gradients for input layer");
        }
        if (layerIdx > this.network.layers.length || layerIdx < 0) {
            throw new Error("Invalid Layer idx");
        }
        
        const previousLayer = this.network.layers[layerIdx - 1];
        const learningData = this.layerConnectionLearningData[layerIdx];        
        const currChainRuleResults = learningData.chainRuleResultByNode;
        const weights = this.network.weights[layerIdx];
        const noOfInputNodes = weights.n;
        const noOfOutputNodes = weights.m;
        for (let outputNodeIdx = 0; outputNodeIdx < noOfOutputNodes; outputNodeIdx++) {
            const outputNodeChainRuleResult = currChainRuleResults.getValue(outputNodeIdx, 0);
            for (let inputNodeIdx = 0; inputNodeIdx < noOfInputNodes; inputNodeIdx++) {
                const inputNodeValue = previousLayer.values().getValue(inputNodeIdx, 0);                
                const derivativeCostWrtWeight = inputNodeValue * outputNodeChainRuleResult;
                const currentGradient = learningData.incomingWeightCostGradient.getValue(outputNodeIdx, 0) as number;
                const newGradient = currentGradient + derivativeCostWrtWeight;
                learningData.incomingWeightCostGradient.setValue(outputNodeIdx, 0, newGradient);
            }
            const derivativeCostWrtBias = 1 * outputNodeChainRuleResult;
            const currBiasGradient = learningData.biasGradientByNode.getValue(outputNodeIdx, 0) as number;
            const newBiasGradient = currBiasGradient + derivativeCostWrtBias;
            learningData.biasGradientByNode.setValue(outputNodeIdx, 0, newBiasGradient);

        }
    }

    protected updateAllGradients(dataPoint: LabelledDataPoint<Dimension, Dimension>): void {
        this.makeForwardPass(dataPoint);
        const outputLayerIdx = this.network.layers.length - 1;
        this.setOutputLayerChainRuleResult(dataPoint);
        this.updateGradients(outputLayerIdx);
        for (let hiddenLayerIdx = outputLayerIdx - 1; hiddenLayerIdx >= 0; hiddenLayerIdx--) {
            this.setHiddenLayerChainRuleResult(hiddenLayerIdx);
            this.updateGradients(hiddenLayerIdx);
        }
    }

    protected applyAllGradients(learningRate: number, momentum: number = 0, regularization: number = 0): void {
        for (let layerIdx = 0; layerIdx < this.network.layers.length - 1; layerIdx++) {
            this.applyGradients(layerIdx, learningRate, momentum, regularization);
        }
    }

    protected applyGradients(layerIdx: number, learningRate: number, momentum: number = 0, regularization: number = 0): void {
        if (layerIdx < 0 || layerIdx >= this.network.layers.length - 1) {
            throw new Error("Invalid layer idx");
        }
        const weightDecay = (1 - learningRate * regularization);
        const nextLayerLearningData = this.layerConnectionLearningData[layerIdx + 1];
        const nextLayerWeightVelocities = nextLayerLearningData.weightVelocities;
        const nextLayerBiasVelocities = nextLayerLearningData.biasVelocities;
        const nextLayerCostGradients = nextLayerLearningData.incomingWeightCostGradient;
        const nextLayerBiasGradients = nextLayerLearningData.biasGradientByNode;
        const weightsToNextLayer = this.network.weights[layerIdx + 1];
        const nextLayerBiases = this.network.layers[layerIdx + 1].biases();

        for (let i = 0; i < weightsToNextLayer.n; i++) {
            for (let j = 0; j < weightsToNextLayer.m; j++) {
                const weight = weightsToNextLayer.getValue(i, j);
                const outputCostGradient = nextLayerCostGradients.getValue(i, 0);
                const weightVelocity = nextLayerWeightVelocities.getValue(i, 0);
                const newWeightVelocity = (momentum * weightVelocity) - (learningRate * (outputCostGradient as number)); //@TODO: Fix costgradient-type
                nextLayerWeightVelocities.setValue(i, 0, newWeightVelocity);
                const newWeight = weightDecay * weight + newWeightVelocity;
                weightsToNextLayer.setValue(i, j, newWeight);
                nextLayerCostGradients.setValue(i, 0, 0);
            }
        }

        for (let i = 0; i < nextLayerBiasGradients.n; i++) {
            const biasVelocity = nextLayerBiasVelocities.getValue(i, 0);
            const outputCostGradient = nextLayerBiasGradients.getValue(i, 0);
            const newBiasVelocity = (momentum * biasVelocity) - (learningRate * (outputCostGradient as number));
            nextLayerBiasVelocities.setValue(i, 0, newBiasVelocity);
            const currBias = nextLayerBiases.getValue(i, 0);
            const newBias = currBias + newBiasVelocity;
            nextLayerBiases.setValue(i, 0, newBias);
            nextLayerBiasGradients.setValue(i, 0, 0);
        }
    }

    learnBatch(dataPoints: Array<LabelledDataPoint<Dimension, Dimension>>, learningRate: number, momentum: number = 0, regularization: number = 0): void {
        for (const dataPoint of dataPoints) {
            this.updateAllGradients(dataPoint);
        }
        this.applyAllGradients(learningRate / dataPoints.length, momentum, regularization);
    }

    train<N extends number, M extends number>(dataPointGenerator: LabelledDataPointGenerator<N, M>, learningRate: number, momentum: number, regularization: number, batchSize: number, epochs: number): void {
        for (let layerIdx = 1; layerIdx < this.network.layers.length; layerIdx++) {
            this.randomizeWeights(layerIdx);
        }

        if (batchSize === 0) {
            const batch = Array.from(dataPointGenerator);
            this.learnBatch(batch, learningRate, momentum, regularization);
            return
        }

        const batchGenerator = function*(dataPointGenerator: LabelledDataPointGenerator<N, M>, batchSize: number) {
            let newBatch = [];
            let j = 0;
            let curr = dataPointGenerator.next();
            while (!curr.done) {
                newBatch.push(curr.value)
                if (newBatch.length === batchSize) {
                    yield newBatch;
                    newBatch = [];
                    j = 0;
                }
                j++;
            }
            return newBatch;
        }(dataPointGenerator, batchSize);

        
        let curr = batchGenerator.next();
        while (!curr.done) {
            this.learnBatch(curr.value, learningRate, momentum, regularization);
        }
        return;
    }



}

class Backpropagation<O extends Dimension> {

    protected layers: Layer<Dimension, number>[];
    protected weights: Record<number, ConnectionMatrix<Dimension, Dimension, number>>;
    protected intermediateValues: Record<number, MutableMatrix<Dimension, 1, number>>;
    
    protected costGradientWeightsByTargetLayer: Record<number, ConnectionMatrix<Dimension, Dimension, number>> = {};
    protected costGradientBiasesByOutputLayer: Record<number, MutableColumn<Dimension, number>> = {};

    protected applyGradients<N extends Dimension, M extends Dimension>(
        layer: Layer<N, number>, 
        incomingWeights: ConnectionMatrix<N, M, number>,
        incomingWeightGradiants: ConnectionMatrix<N, M, number>, 
        biasGradients: MutableColumn<N, number>, 
        learningRate: number
    ) {
        const biases = layer.biases();
        for (let i = 0; i < biases.n; i++) {
            biases.setValue(i, 0, biases.getValue(i, 0) - learningRate * biasGradients.getValue(i, 0));
            for (let j = 0; j < incomingWeights.m; j++) {
                incomingWeights.setValue(i, j, incomingWeights.getValue(i, j) - learningRate * incomingWeightGradiants.getValue(i, j));
            }
        }        
    }

    protected calculateOutputLayerNodeDerivatives<O extends Dimension>(expectedOutputs: Column<O, number>, learningError: LearningError<O>): void  {
        const outputLayer = this.layers[this.layers.length - 1];
        const outputLayerValues = outputLayer.activations();
        const outputLayerBiases = outputLayer.biases();
        const activationFunctionClass = outputLayer.activationFunction();
        const activationFunction = activationFunctionClass.calculate;
        const activationFunctionDerivative = activationFunctionClass.derivative;

    }

    protected applyAllGradients() {
        for (let i = this.layers.length - 1; i > 0; i--) {
            const layer = this.layers[i];
            const incomingWeights = this.weights[i];
            const incomingWeightGradiants = this.costGradientWeightsByTargetLayer[i];
            const biasGradients = this.costGradientBiasesByOutputLayer[i];
            if (incomingWeightGradiants === undefined || biasGradients === undefined) {
              this.applyGradients(layer, incomingWeights, incomingWeightGradiants, biasGradients, 0.1);
            }
        }
    }

    constructor(public readonly initialLayerRef: Layer<Dimension, number>, learningError: LearningError<O>) {
        

        let currLayer: Layer<Dimension, number> = initialLayer;
        let nextLayer: Layer<Dimension, number> = currLayer.nextLayer();
        this.layers.push(currLayer);

        let i = 1;
        while (nextLayer !== undefined) {
            this.layers[i] = nextLayer;
            this.weights[1 + 1] = currLayer.nextLayerConnection();
            currLayer = nextLayer;
            nextLayer = currLayer.nextLayer();
            i++;
        }



    }

    
}

            