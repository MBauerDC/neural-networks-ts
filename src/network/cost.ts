import { Column, Dimension } from "../../node_modules/matrices/src/matrix";
export type TotalCostEvaluator<N extends Dimension> = (distance: Column<N, number>) => number;
export type CostFunction<N extends Dimension> = (expected: Column<N, number>, actual: Column<N, number>) => number;
export type CostFunctionCtor<N extends Dimension> = (nodeCostFunction: NodeCostFunction) => CostFunction<N>;
export type NodeCostFunction = (expected: number, actual: number) => number;
export type NodeCostFunctionDerivative = (expected: number, actual: number) => number;

export const meanNodeCostEvaluator: TotalCostEvaluator<Dimension> = (distance: Column<Dimension, number>) => { 
    return sumNodeCostEvaluator(distance) / distance.n;
}

export const sumNodeCostEvaluator: TotalCostEvaluator<Dimension> = (distance: Column<Dimension, number>) => {
    return Array.from(distance.generateColumn(0)).reduce((a, b) => a + b, 0);
}

export const squaredError: NodeCostFunction = (expected, actual) => Math.pow(expected - actual, 2);
export const squaredErrorDerivative: NodeCostFunctionDerivative = (expected, actual) => 2 * (actual - expected);
export const binaryCrossEntropy: NodeCostFunction = (expected, actual) => -expected * Math.log(actual) - (1 - expected) * Math.log(1 - actual);
export const binaryCrossEntropyDerivative: NodeCostFunctionDerivative = (expected, actual) => (actual - expected) / (actual * (1 - actual));
export const categoricalCrossEntropy: NodeCostFunction = (expected, actual) => -expected * Math.log(actual);
export const categoricalCrossEntropyDerivative: NodeCostFunctionDerivative = (expected, actual) => -expected / actual;
export const getLinearCostFunction: CostFunctionCtor<Dimension> = <N extends Dimension>(nodeCostFunction: NodeCostFunction) => {
  const vectorFn: CostFunction<N> = (expected: Column<N, number>, actual: Column<N, number>) => {
    return Array.from(expected.generateColumn(0)).reduce((acc, expectedValue, i) => {
      acc += nodeCostFunction(expectedValue, actual.getValue(i, 0));
      return acc;
    });
  };
  return vectorFn;
}
export const getMeanCostFunction: CostFunctionCtor<Dimension> = <N extends Dimension>(nodeCostFunction: NodeCostFunction) => {
    const vectorFn: CostFunction<N> = (expected: Column<N, number>, actual: Column<N, number>) => {
        return Array.from(expected.generateColumn(0)).reduce((acc, expectedValue, i) => {
            acc += nodeCostFunction(expectedValue, actual.getValue(i, 0));
            return acc;
        }) / expected.n;
    };
    return vectorFn;
}

export type LearningError<N extends Dimension> = { 
    nodeCostFunction: NodeCostFunction, 
    nodeCostFunctionDerivative: NodeCostFunctionDerivative, 
    costFunction: CostFunction<N>, 
    totalCostEvaluator: TotalCostEvaluator<N>,
    distance(expected: Column<N, number>, actual: Column<N, number>): Column<N, number>, 
    nodeCost(expected: Column<N, number>, actual: Column<N, number>, i: number): number,
    cost(expected: Column<N, number>, actual: Column<N, number>): number,
    distanceCost(distance: Column<N, number>): number,
}

export type LearningErrorCtor<N extends Dimension> = 
  (nodeCostFunction: NodeCostFunction, nodeCostFunctionDerivative: NodeCostFunctionDerivative, costFunctionCtor: CostFunctionCtor<N>, totalCostEvaluator: TotalCostEvaluator<N>) => 
    LearningError<N>;

export const learningErrorCtor: LearningErrorCtor<Dimension> = <N extends Dimension>(nodeCostFunction, nodeCostFunctionDerivative, costFunctionCtor, totalCostEvaluator) => {
    return new class {
      public readonly costFunction: CostFunction<N>;  
      constructor(
        public readonly nodeCostFunction: NodeCostFunction, 
        public readonly nodeCostFunctionDerivative: NodeCostFunctionDerivative, 
        costFunctionCtor: CostFunctionCtor<N>,
        public readonly totalCostEvaluator: TotalCostEvaluator<N>,
      ) {
        this.costFunction = costFunctionCtor(nodeCostFunction);
      }
      distance(expected: Column<N, number>, actual: Column<N, number>): Column<N, number> {
        return expected.withSubtracted(actual) as Column<N, number>;
      }
      nodeCost(expected: Column<N, number>, actual: Column<N, number>, i: number): number {
        if (i < 0 || i >= expected.n) {
            throw new Error(`Index ${i} is out of bounds for column of length ${expected.n}`);
        }
        return this.nodeCostFunction(expected.getValue(i, 0), actual.getValue(i, 0));
      }
      cost(expected: Column<N, number>, actual: Column<N, number>): number {
        return this.costFunction(expected, actual);
      }
      distanceCost(distance: Column<N, number>): number {
        return this.totalCostEvaluator(distance);
      }
    }(nodeCostFunction, nodeCostFunctionDerivative, costFunctionCtor, totalCostEvaluator);
};

export const meanSquaredLearningError = learningErrorCtor(squaredError, squaredErrorDerivative, getMeanCostFunction, meanNodeCostEvaluator);
