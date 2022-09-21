import { Dimension, Matrix } from "../../node_modules/matrices/src/matrix";
import { LearningError } from "./cost";
import { EncodingDecodingScheme } from "./encoding";

type LabelType = string|number;

interface ProblemSpecification<I extends Dimension, O extends Dimension> {
    i: I,
    o: O,
    isClassificationProblem: boolean;
    isRegressionProblem: boolean;
    error: LearningError<O>;
  }
  

interface ClassificationProblem<I extends Dimension, O extends Dimension, T extends LabelType> extends ProblemSpecification<I, O> {
    isClassificationProblem: true;
    isRegressionProblem: false;
    classes: number[];
    classLabels: string[];
    encodingDecodingScheme: EncodingDecodingScheme<O, T, number>;
}

interface RegressionProblem<N extends Dimension, M extends Dimension> extends ProblemSpecification<N, M> {
    isClassificationProblem: false;
    isRegressionProblem: true;
}

interface LearningProblemDataSequenceResult<I extends Dimension, O extends Dimension> {
    error: number[];
    averageError(): number;
}

interface RegressionProblemSinglePassResult<I extends Dimension, O extends Dimension> extends LearningProblemDataSequenceResult<I, O> {};
interface ClassificationProblemDataSequenceResult<I extends Dimension, O extends Dimension, T extends LabelType> extends LearningProblemDataSequenceResult<I, O> {
    confusionMatrix: Matrix<O, O, number>;
    noOfTruePositives: number;
    noOfTrueNegatives: number;
    noOfFalsePositives: number;
    noOfFalseNegatives: number;
    accuracy(): number;
    precision(): number;
    recall(): number;
    f1Score(): number;
}

interface MultiClassClassificationProblemDataSequenceResult<I extends Dimension, O extends Dimension, T extends LabelType> extends ClassificationProblemDataSequenceResult<I, O, T> {
    forLabelIdx: number;
}

interface LearningProblemTestResult<I extends Dimension, O extends Dimension> {
    averageError: number;
}

class GenericLearningProblemTestResult<I extends Dimension, O extends Dimension> implements LearningProblemTestResult<I, O> {
    public readonly averageError: number;
    constructor(sequenceResults: LearningProblemDataSequenceResult<I, O>[]) {
        this.averageError = sequenceResults.map(r => r.averageError()).reduce((a, b) => a + b, 0) / sequenceResults.length;
    }
}

interface ClassificationProblemTestResult<I extends Dimension, O extends Dimension, T extends LabelType> extends LearningProblemTestResult<I, O> {
    confusionMatrix: Matrix<O, O, number>;
    averageError: number;
    averageAccuracy: number;
    averagePrecision: number;
    averageRecall: number;
    averageF1Score: number;
}

class GenericClassificationProblemTestResult<I extends Dimension, O extends Dimension, T extends LabelType> implements ClassificationProblemTestResult<I, O, T> {
    public readonly confusionMatrix: Matrix<O, O, number>;
    public readonly averageError: number;
    public readonly averageAccuracy: number;
    public readonly averagePrecision: number;
    public readonly averageRecall: number;
    public readonly averageF1Score: number;
    constructor(sequenceResults: ClassificationProblemDataSequenceResult<I, O, T>[]) {
        this.confusionMatrix = sequenceResults.map(r => r.confusionMatrix).reduce((a, b) => a.withAdded(b));
        this.averageError = sequenceResults.map(r => r.averageError()).reduce((a, b) => a + b, 0) / sequenceResults.length;
        this.averageAccuracy = sequenceResults.map(r => r.accuracy()).reduce((a, b) => a + b, 0) / sequenceResults.length;
        this.averagePrecision = sequenceResults.map(r => r.precision()).reduce((a, b) => a + b, 0) / sequenceResults.length;
        this.averageRecall = sequenceResults.map(r => r.recall()).reduce((a, b) => a + b, 0) / sequenceResults.length;
        this.averageF1Score = sequenceResults.map(r => r.f1Score()).reduce((a, b) => a + b, 0) / sequenceResults.length;
    }
}

export { LabelType, ProblemSpecification, ClassificationProblem, RegressionProblem, LearningProblemDataSequenceResult, RegressionProblemSinglePassResult, ClassificationProblemDataSequenceResult, MultiClassClassificationProblemDataSequenceResult, LearningProblemTestResult, GenericLearningProblemTestResult, ClassificationProblemTestResult, GenericClassificationProblemTestResult };