import { KeyExportOptions } from "crypto";
import { Domain } from "domain";
import { Column, Dimension, Matrix, MutableColumn, MutableMatrix } from "../matrices/matrix";
import { Network } from "./network";

interface LayerGradientData<N extends Dimension, M extends Dimension> {
    getIncomingWeights(): MutableMatrix<N, M, number>;
    getIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number;
    setIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void;
    getBias(thisLayerNodeIndex: number): number;
    setBias(thisLayerNodeIndex: number, value: number): void;
    getErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number): number;
    setErrorGradientForIncomingWeight(thisLayerNodeIndex: number, previousLayerNodeIndex: number, value: number): void;
    getErrorGradientForBias(nodeIndex: number): number;
    setErrorGradientForBias(nodeIndex: number, value: number): void;
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
    updateWeightsAndBiases<N extends Dimension, M extends Dimension, G extends LayerGradientData<N, M>>(layerGradientData: G): void;

    updateGradients<K extends Dimension, L extends Dimension, C extends Column<K, number>, G extends LayerGradientData<L, K>, D extends Column<L, number>>(previousLayerActivation: C, currentLayerGradientData: G, outgoingNodeCostDifferentials: D): void;
    updateAllGradients(previousLayerActivations: Column<Dimension,  number>[], layerGradientData: LayerGradientData<Dimension, Dimension>[], nodeCostDifferentials: Column<Dimension, number>[]): void;
}

class SimpleGradientDescent<N extends Dimension, M extends Dimension> implements GradientDescentAlgorithm<GradientDescentOptions> {
    constructor(public options: GradientDescentOptions) {}

    updateWeightsAndBiases<N extends number, M extends number>(layerGradientData: LayerGradientData<N, M>): void {
        const weights = layerGradientData.getIncomingWeights();
        for (let i = 0; i < weights.n ; i++) {
            for (let j = 0; j < weights.m; j++) {
                const weight = layerGradientData.getIncomingWeight(i, j);
                const weightGradient = layerGradientData.getErrorGradientForIncomingWeight(i, j);
                const newWeight = weight - this.options.learningRate * weightGradient;
                layerGradientData.setIncomingWeight(i, j, newWeight);
            }
        }

        for (let i = 0; i < weights.n; i++) {
            const bias = layerGradientData.getBias(i);
            const biasGradient = layerGradientData.getErrorGradientForBias(i);
            const newBias = bias - this.options.learningRate * biasGradient;
            layerGradientData.setBias(i, newBias);
        }  
    }

    updateGradients<K extends number, L extends number, C extends Column<K, number>, G extends LayerGradientData<L, K>, D extends Column<L, number>>(previousLayerActivation: C, currentLayerGradientData: G, outgoingNodeCostDifferentials: D): void {
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
    updateAllGradients(previousLayerActivations: Column<number, number>[], layerGradientData: LayerGradientData<number, number>[], nodeCostDifferentials: Column<number, number>[]): void {
        if (previousLayerActivations.length !== layerGradientData.length || nodeCostDifferentials.length !== layerGradientData.length) {
            throw new Error("previousLayerActivations.length !== layerGradientData.length - 1 || nodeCostDifferentials.length !== layerGradientData.length - 1");
        }
        for (let i = 0; i < layerGradientData.length; i++) {
            this.updateGradients(previousLayerActivations[i], layerGradientData[i], nodeCostDifferentials[i]);
        }
    }
}

