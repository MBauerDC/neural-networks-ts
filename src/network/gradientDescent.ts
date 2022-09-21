import { Column, Dimension, GenericColumn, GenericMutableColumn, GenericMutableMatrix, Matrix, MutableColumn, MutableMatrix, toMutable } from "../../node_modules/matrices/src/index";
import { initializeWeightMatrix } from "../utils/random";
import { LearningError } from "./cost";
import { ProblemSpecification } from "./learningProblem";
import { Network, LabelledDataPointGenerator, LabelledDataPoint } from "./network";

interface LayerGradientData<N extends Dimension, M extends Dimension> {
    getIncomingWeights(): MutableMatrix<N, M, number>;
    getIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number;
    setIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void;
    getBias(thisLayerNodeIndex: number): number;
    setBias(thisLayerNodeIndex: number, value: number): void;
    getErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number;
    setErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void;
    setErrorGradientsForIncomingWeights(gradients: Matrix<N, M, number>): void;
    getErrorGradientForBias(thisLayerNodeIndex: number): number;
    setErrorGradientForBias(thisLayerNodeIndex: number, value: number): void;
    setErrorGradientsForBiases(gradients: Column<N, number>): void;
    getErrorGradientsForIncomingWeights(): MutableMatrix<N, M, number>;
    getErrorGradientsForBiases(): MutableColumn<N, number>;
    getNodeCostDerivative(nodeIndex: number): number;
    setNodeCostDerivative(nodeIndex: number, value: number): void;
    getNodeCostDerivatives(): MutableColumn<N, number>;
    setNodeCostDerivatives(derivatives: Column<N, number>): void;
}

interface LayerGradientDataFactory<N extends Dimension, M extends Dimension> {
    createLayerGradientData(thisLayerNodeCount: number, previousLayerNodeCount: number): LayerGradientData<N, M>;
}

interface LayerGradientDataFactoryImpl<N extends Dimension, M extends Dimension> extends LayerGradientDataFactory<N, M> {
    createLayerGradientData(thisLayerNodeCount: number, previousLayerNodeCount: number): LayerGradientDataImpl<N, M>;
}

class LayerGradientDataImpl<N extends Dimension, M extends Dimension> implements LayerGradientData<N, M> {
    private incomingWeights: MutableMatrix<N, M, number>;
    private incomingWeightErrorGradients: MutableMatrix<N, M, number>;
    private biases: MutableColumn<N, number>;
    private biasErrorGradients: MutableColumn<N, number>;
    private nodeCodeDerivatives: MutableColumn<N, number>;
    constructor(thisLayerNodeCount: number, previousLayerNodeCount: number) {
        this.incomingWeights = new GenericMutableMatrix<N, M, number>(Array<Array<number>>(thisLayerNodeCount).fill(Array<number>(previousLayerNodeCount).fill(0)), null, null, thisLayerNodeCount as N, previousLayerNodeCount as M);
        this.incomingWeightErrorGradients = new GenericMutableMatrix<N, M, number>(Array<Array<number>>(thisLayerNodeCount).fill(Array<number>(previousLayerNodeCount).fill(1)), null, null, thisLayerNodeCount as N, previousLayerNodeCount as M);
        this.biases = new GenericMutableColumn<N, number>(Array(thisLayerNodeCount).fill(0), thisLayerNodeCount as N);
        this.biasErrorGradients = new GenericMutableColumn<N, number>(Array(thisLayerNodeCount).fill(0), thisLayerNodeCount as N);
        this.nodeCodeDerivatives = new GenericMutableColumn<N, number>(Array(thisLayerNodeCount).fill(1), thisLayerNodeCount as N);
    }
    getIncomingWeights(): MutableMatrix<N, M, number> {
        return this.incomingWeights;
    }
    getIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number {
        return this.incomingWeights.getValue(thisLayerNodeIndex, previousLayerNodeIndex);
    }
    setIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void {
        this.incomingWeights.setValue(thisLayerNodeIndex, previousLayerNodeIndex, value);
    }
    getBias(thisLayerNodeIndex: number): number {
        return this.biases.at(thisLayerNodeIndex);
    }
    setBias(thisLayerNodeIndex: number, value: number): void {
        this.biases.setValue(thisLayerNodeIndex, 0, value);
    }
    getErrorGradientForBias(nodeIndex: number): number {
        return this.biasErrorGradients.at(nodeIndex);
    }
    getErrorGradientsForBiases(): MutableColumn<N, number> {
        return this.biasErrorGradients;
    }
    setErrorGradientForBias(nodeIndex: number, value: number): void {
        this.biasErrorGradients.setValue(nodeIndex, 0, value);
    }
    getErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number {
        return this.incomingWeightErrorGradients.getValue(thisLayerNodeIndex, previousLayerNodeIndex);
    }
    getErrorGradientsForIncomingWeights(): MutableMatrix<N, M, number> {
        return this.incomingWeightErrorGradients;
    }
    setErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void {
        this.incomingWeightErrorGradients.setValue(thisLayerNodeIndex, previousLayerNodeIndex, value);
    }
    setErrorGradientsForIncomingWeights(gradients: Matrix<N, M, number>): void {
        this.incomingWeightErrorGradients = toMutable(gradients) as MutableMatrix<N, M, number>;
    }
    setErrorGradientsForBiases(gradients: Column<N, number>): void {
        this.biasErrorGradients = toMutable(gradients) as MutableColumn<N, number>;
    }
    getNodeCostDerivative(nodeIndex: number): number {
        return this.nodeCodeDerivatives.at(nodeIndex);
    }
    setNodeCostDerivative(nodeIndex: number, value: number): void {
        this.nodeCodeDerivatives.setValue(nodeIndex, 0, value);
    }
    getNodeCostDerivatives(): MutableColumn<N, number> {
        return this.nodeCodeDerivatives;
    }
    setNodeCostDerivatives(derivatives: Column<N, number>): void {
        this.nodeCodeDerivatives = toMutable(derivatives) as MutableColumn<N, number>;
    }
}

interface LayerGradientDataWithHistory<N extends Dimension, M extends Dimension> extends LayerGradientData<N, M> {
    getIncomingWeightHistory(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number[];
    getErrorGradientHistoryForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number[];
    getErrorGradientHistoryForBias(nodeIndex: number): number[];
}

interface LayerGradientDataWithVelocity<N extends Dimension, M extends Dimension> extends LayerGradientData<N, M> {
    getIncomingWeightVelocities(): MutableMatrix<N, M, number>;
    getVelocityForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number;
    setVelocityForIncomingWeight(nodeIndex: number, value: number): number;
    getBiasVelocities(): MutableMatrix<N, 1, number>;
    getVelocityForBias(nodeIndex: number): number;
    setVelocityForBias(nodeIndex: number, value: number): number;
}

interface LayerGradientDataWithVelocityAndHistory<N extends Dimension, M extends Dimension> extends LayerGradientDataWithHistory<N, M>, LayerGradientDataWithVelocity<N, M> {
    getVelocityHistoryForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number[];
    getVelocityHistoryForBias(nodeIndex: number): number[];
}


interface GradientDescentOptions {
    learningRate: number;
}

interface GradientDescentOptionsWithMomentum extends GradientDescentOptions {
    momentum: number;
}

interface GradientDescentOptionsWithRegularization extends GradientDescentOptions {
    regularizationFactor: number;
}

interface GradientDescentAlgorithm<O extends GradientDescentOptions> {
    options: O;
    updateWeightsAndBiases<N extends Dimension, M extends Dimension, G extends LayerGradientData<N, M>>(layerGradientData: G, optionOverride: O|null): void;

    updateGradients<K extends Dimension, L extends Dimension, C extends Column<K, number>, G extends LayerGradientData<L, K>, D extends Column<L, number>>(previousLayerActivation: C, currentLayerGradientData: G, outgoingNodeCostDifferentials: D, optionOverride: O|null): void;
    updateAllGradients(previousLayerActivations: Column<Dimension,  number>[], layerGradientData: LayerGradientData<Dimension, Dimension>[], optionOverride: O|null): void;

    initializeAllLayerGradientData(network: Network<Dimension, Dimension>): LayerGradientData<Dimension, Dimension>[];
}

interface BackpropagationGradientDescentTrainer<P extends GradientDescentOptions, A extends GradientDescentAlgorithm<P>> {
    algorithm: A;
    train<I extends Dimension, O extends Dimension>(
        learningProblem: ProblemSpecification<I, O>, 
        network: Network<I, O>, 
        trainingDataGenerator: LabelledDataPointGenerator<I, O>,
        batchSize: number,
        epochs: number,
        learningError: LearningError<O>,
        optionsOverride: P|null
    ): Network<I, O>
}

class GenericBackpropagationGrandientDescentTrainer<P extends GradientDescentOptions, A extends GradientDescentAlgorithm<P>> implements BackpropagationGradientDescentTrainer<P, A> {
    constructor(public readonly algorithm: A) {
        this.algorithm = algorithm;
    }

    protected getLayerActivations(network: Network<Dimension, Dimension>): MutableColumn<Dimension, number>[] {
        const layerActivations: MutableColumn<Dimension, number>[] = [];
        for (let i = 0; i < network.layers.length - 1; i++) {
            const layer = network.layers[i];
            const layerActivation = layer.activations;
            const layerActivationFirstColumn = layerActivation.getColumn(0);
            layerActivations.push(layerActivationFirstColumn);
        }
        return layerActivations;
    }

    protected setOutputLayerNodeCostDifferential<I extends Dimension, O extends Dimension>(network: Network<I,O>, learningError: LearningError<O>, dataPoint: LabelledDataPoint<I, O>, layerData: LayerGradientData<Dimension, O>): void {
        const outputLayer = network.layers[network.layers.length - 1];
        const nodeCostDerivative = learningError.nodeCostFunctionDerivative;
        for (let i = 0; i < outputLayer.n; i++) {
            const costDerivative = nodeCostDerivative(outputLayer.activations.getValue(i, 0), dataPoint.output.getValue(i, 0));
            const activationDerivative = outputLayer.activationFunction.derivative(outputLayer.getSummedInputs()?.getColumn(0).getValue(i, 0) || 0);
            layerData.setNodeCostDerivative(i, activationDerivative * costDerivative);
        }
    }

    protected setHiddenLayerNodeCostDifferential<N extends Dimension, M extends Dimension>(network: Network<Dimension, Dimension>, layerIndex: number, layerData: LayerGradientData<N, M>, nextLayerData: LayerGradientData<Dimension, N>): void {
        /*const layer = network.layers[layerIndex];
        const nextLayer = network.layers[layerIndex + 1];
        for (let i = 0; i < layer.n; i++) {
            let sum = 0;
            for (let j = 0; j < nextLayer.n; j++) {
                sum += nextLayerData.getIncomingWeight(j, i) * nextLayerData.getNodeCostDerivative(j);
            }
            const activationDerivative = layer.activationFunction.derivative(layer.getSummedInputs()?.getColumn(0).getValue(i, 0) || 0);
            layerData.setNodeCostDerivative(i, activationDerivative * sum);
        }*/
        if (layerIndex === 0) {
            throw new Error("Cannot set chain rule result for input layer");
        }
        if (layerIndex >= network.layers.length - 1) {
            throw new Error("Invalid layer idx.");
        }
        const layer = network.layers[layerIndex];
        const nextLayer = network.layers[layerIndex + 1];
        const outgoingWeights = network.weights[layerIndex + 1];
        const noOfNodesInThisLayer = layer.n;
        const noOfNodesInNextLayer = nextLayer.n;

        for (let thisLayerNodeIndex = 0; thisLayerNodeIndex < noOfNodesInThisLayer; thisLayerNodeIndex++) {
            let newNodeCostDerivative: number = 0;
            // Sum the chain rule results for each connected next layer node
            for (let nextLayerNodeIndex = 0; nextLayerNodeIndex < noOfNodesInNextLayer; nextLayerNodeIndex++) {
                const weightedInputDerivative = outgoingWeights.getValue(nextLayerNodeIndex, thisLayerNodeIndex);
                const nextLayerNodeChainRuleResult = nextLayerData.getNodeCostDerivative(nextLayerNodeIndex);
                newNodeCostDerivative += weightedInputDerivative * nextLayerNodeChainRuleResult;
            }
            // Multiply by the derivative of the activation function evaluated at the current summed input value of the current node
            const currNodeSummedInputValue = layer.getSummedInputs()?.getColumn(0).getValue(thisLayerNodeIndex, 0) || 0;
            newNodeCostDerivative *= layer.activationFunction.derivative(currNodeSummedInputValue);
            layerData.setNodeCostDerivative(thisLayerNodeIndex, newNodeCostDerivative);
        }
    }

    

    protected *batchGenerator<N extends Dimension, M extends Dimension>(dataPointGenerator: LabelledDataPointGenerator<N, M>, batchSize: number) {
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
    }

    protected dataPointBatchToMatrix<I extends Dimension, O extends Dimension, M extends Dimension>(batch: LabelledDataPoint<I, O>[]): [MutableMatrix<I, M, number>, MutableMatrix<O, M, number>] {
        const [listOfInputColumns, listOfOutputColumn] = [
            batch.map((dataPoint) => toMutable(dataPoint.input.getColumn(0)) as MutableColumn<I, number>),
            batch.map((dataPoint) => toMutable(dataPoint.output.getColumn(0)) as MutableColumn<O, number>)
        ];
        return [
            new GenericMutableMatrix(null, null, listOfInputColumns, listOfInputColumns[0].n as I, listOfInputColumns.length as M),
            new GenericMutableMatrix(null, null, listOfOutputColumn, listOfOutputColumn[0].n as O, listOfOutputColumn.length as M)
        ];
    }

    protected calculateCost<O extends Dimension, M extends Dimension>(learningError: LearningError<O>, actualOutput: Matrix<O, M, number>, desiredOutput: Matrix<O, M, number>): number {
        const distance = desiredOutput.withSubtracted(actualOutput);
        let sum = 0;
        for (let j = 0; j < distance.m; j++) {
            const dataPointDistance = distance.getColumn(j);
            const dataPointError = learningError.distanceCost(dataPointDistance);
            sum += dataPointError;
        }
        return sum / distance.m;
    }

    protected randomizeWeights(network: Network<Dimension, Dimension>, outputLayerIdx: number): void {
        if (outputLayerIdx < 1 || outputLayerIdx >= network.layers.length) {
            throw new Error('Invalid layer idx.');
        }
        const outputLayer = network.layers[outputLayerIdx];
        const currWeightsToLayer = network.weights[outputLayerIdx];
        const activationFn = outputLayer.activationFunction;
        const randomizedWeights = initializeWeightMatrix(currWeightsToLayer.n, currWeightsToLayer.m, activationFn)
        network.weights[outputLayerIdx] = randomizedWeights;
    }
    
    protected randomizeAllWeights(network: Network<Dimension, Dimension>): void {
        for (let i = 1; i <= network.layers.length - 1; i++) {
            this.randomizeWeights(network, i);
        }
    }

    protected learnBatch<I extends Dimension, O extends Dimension>(
        network: Network<I, O>, 
        layerGradientData: LayerGradientData<Dimension, Dimension>[], 
        errorFunction: LearningError<O>, 
        layerActivations: Column<Dimension, number>[], 
        optionsOverride: P|null, 
        batch: LabelledDataPoint<I, O>[]
    ): void {
        const gradientsToAverageByLayerIdx: Record<number, [Column<Dimension, number>[], Matrix<Dimension, Dimension, number>[]]> = {};
        for (const dataPoint of batch) {
            this.setOutputLayerNodeCostDifferential(network, errorFunction as unknown as LearningError<number>, dataPoint, layerGradientData[layerGradientData.length - 1]);
            for (let i = network.layers.length - 2; i > 0; i--) {
                this.setHiddenLayerNodeCostDifferential(network, i, layerGradientData[i], layerGradientData[i + 1]);
            }
            this.algorithm.updateAllGradients(layerActivations as Column<Dimension, number>[], layerGradientData, optionsOverride);
            for (let i = 1; i <= network.layers.length - 1; i++) {
                const currGradientData = layerGradientData[i];
                const newGradients = gradientsToAverageByLayerIdx[i] || [[], []];
                newGradients[0].push(currGradientData.getErrorGradientsForBiases() as Column<Dimension, number>);
                newGradients[1].push(currGradientData.getErrorGradientsForIncomingWeights() as Matrix<Dimension, Dimension, number>);
                gradientsToAverageByLayerIdx[i] = newGradients;
            }
        }
        for (let i = network.layers.length - 1; i > 0; i--) {
            // Average the gradients
            const [biasGradients, weightGradients] = gradientsToAverageByLayerIdx[i];
            let biasColumnAvg = new GenericMutableColumn(Array(biasGradients[0].n).fill(0), biasGradients[0].n);
            let weightMatrixAvg = new GenericMutableMatrix<Dimension, Dimension, number>(Array(weightGradients[0].n).fill(Array(weightGradients[0].m).fill(0)), null, null, weightGradients[0].n, weightGradients[0].m);
            const batchSize = batch.length;
            for (let b = 0; b < batchSize; b++) {
                biasColumnAvg.add(biasGradients[b]);
                weightMatrixAvg.add(weightGradients[b]);
            }
            biasColumnAvg.mapInPlace((x) => x / batchSize);
            weightMatrixAvg.mapInPlace((x) => x / batchSize);
            layerGradientData[i - 1].setErrorGradientsForBiases(biasColumnAvg as Column<Dimension, number>);
            layerGradientData[i - 1].setErrorGradientsForIncomingWeights(weightMatrixAvg as Matrix<Dimension, Dimension, number>);
            this.algorithm.updateWeightsAndBiases(layerGradientData[i - 1], optionsOverride);
        }
    };


    train<I extends Dimension, O extends Dimension>(
        learningProblem: ProblemSpecification<I, O>, 
        network: Network<I, O>, 
        trainingDataGenerator: LabelledDataPointGenerator<I, O>,
        batchSize: number,
        epochs: number,
        errorFunction: LearningError<O>,
        optionsOverride: P|null
    ): Network<I, O> {
        const layerGradientData = this.algorithm.initializeAllLayerGradientData(network);
        const layerActivations = this.getLayerActivations(network);
        
        this.randomizeAllWeights(network);
        for (let e = 0; e < epochs; e++) {
            const currBatchIt = this.batchGenerator(trainingDataGenerator, batchSize);
            let currBatchItVal = currBatchIt.next();
            while (!currBatchItVal.done) {
                this.learnBatch(
                    network,
                    layerGradientData,
                    errorFunction,
                    layerActivations as Column<Dimension, number>[],
                    optionsOverride,
                    currBatchItVal.value
                );
            }
        }

        return network;
    }
}

class SimpleGradientDescent<N extends Dimension, M extends Dimension> implements GradientDescentAlgorithm<GradientDescentOptions> {
    constructor(public options: GradientDescentOptions) {}

    initializeAllLayerGradientData(network: Network<number, number>): LayerGradientData<number, number>[] {
        const layerGradientData: LayerGradientData<number, number>[] = [];
        for (let i = 1; i <= network.layers.length - 1; i++) {
            layerGradientData.push(new LayerGradientDataImpl(
                network.layers[i].n,
                network.layers[i - 1].n,
            ));
        }
        return layerGradientData;
    }

    updateWeightsAndBiases<N extends number, M extends number>(layerGradientData: LayerGradientData<N, M>, optionOverride: GradientDescentOptions|null = null): void {
        const weights = layerGradientData.getIncomingWeights();
        const learningRate = optionOverride?.learningRate ?? this.options.learningRate;
        for (let i = 0; i < weights.n ; i++) {
            for (let j = 0; j < weights.m; j++) {
                const weight = layerGradientData.getIncomingWeight(i, j);
                const weightGradient = layerGradientData.getErrorGradientForIncomingWeight(i, j);
                const newWeight = weight - learningRate * weightGradient;
                layerGradientData.setIncomingWeight(i, j, newWeight);
            }
        }

        for (let i = 0; i < weights.n; i++) {
            const bias = layerGradientData.getBias(i);
            const biasGradient = layerGradientData.getErrorGradientForBias(i);
            const newBias = bias - learningRate * biasGradient;
            layerGradientData.setBias(i, newBias);
        }  
    }

    updateGradients<K extends number, L extends number, C extends Column<K, number>, G extends LayerGradientData<L, K>, D extends Column<L, number>>(previousLayerActivation: C, currentLayerGradientData: G, outgoingNodeCostDifferentials: D, optionOverride: GradientDescentOptions|null = null): void {
      const weights = currentLayerGradientData.getIncomingWeights();
      for (let i = 0; i < weights.n; i++) {
        const outgoingNodeCostDifferential = outgoingNodeCostDifferentials.getValue(i, 0);
        for (let j = 0; j < weights.m; j++) {   
          const currErrorGradient = currentLayerGradientData.getErrorGradientForIncomingWeight(i, j);             
          const partialDiffOfSummedBiasedInputWrtWeight = previousLayerActivation.getValue(j, 0);                
          const gradientDiff = outgoingNodeCostDifferential * partialDiffOfSummedBiasedInputWrtWeight;
          currentLayerGradientData.setErrorGradientForIncomingWeight(i, j, currErrorGradient + gradientDiff);
        }
        const biasGradient = currentLayerGradientData.getErrorGradientForBias(i);
        const biasGradientDiff = outgoingNodeCostDifferential * 1;
        currentLayerGradientData.setErrorGradientForBias(i, biasGradient + biasGradientDiff);
      }
    }

    // Arrays must all have length equal to the number of layers in the network minus one (they are layers 2 through n)
    updateAllGradients(previousLayerActivations: Column<number, number>[], layerGradientData: LayerGradientData<number, number>[], optionOverride: GradientDescentOptions|null = null): void {
        const nodeCostDifferentials: MutableColumn<number, number>[] = layerGradientData.map(d => d.getNodeCostDerivatives());
        if (previousLayerActivations.length !== layerGradientData.length || nodeCostDifferentials.length !== layerGradientData.length) {
            throw new Error("previousLayerActivations.length !== layerGradientData.length - 1 || nodeCostDifferentials.length !== layerGradientData.length - 1");
        }
        for (let i = 0; i < layerGradientData.length; i++) {
            this.updateGradients(previousLayerActivations[i], layerGradientData[i], nodeCostDifferentials[i] as Column<Dimension, number>);
        }
    }
}

