import { LinearAlgebra } from "./linearAlgebra";
import { Matrix } from "./matrix";
import { GenericMutableMatrix, MutableMatrix } from "./mutable";


type T = number;
const startConstruct = Date.now();
const threeByTwo: Matrix<3,2,T> = new GenericMutableMatrix<3, 2, T>([[1, 2], [-3, 0], [5, -6]], null, null, 3, 2);
const twoByFour: MutableMatrix<2,4,T> = new GenericMutableMatrix<2, 4, T>([[12, 2, -3, 4], [0, -6, 7, 8]], null, null, 2, 4);
const threeByFour: MutableMatrix<3,4,T> = new GenericMutableMatrix<3, 4, T>([[32, -1, -3, 4], [0, -6, 7, 8], [4, 7, 2.5, 0]], null, null, 3, 4);
const fourByFive: MutableMatrix<4,5,T> = new GenericMutableMatrix<4, 5, T>([[1, 2, 3, -4, 5], [6, 0, 8, 9, 10], [11, 12.5, 13, 14, 15], [16, 17, 18, -19, 20]], null, null, 4, 5);
const swapRowTwoAndThreeOfFour = LinearAlgebra.ElementaryRowOperations.getSwapOperation(1, 2, 4);
const addTwiceRowFourToRowThree = LinearAlgebra.ElementaryRowOperations.getAddScaledOperation(2, 3, 2, 4);
const fourByFiveSwapped: MutableMatrix<4,5,T> = swapRowTwoAndThreeOfFour.getTransformationMatrix().getMultiplication(fourByFive);
const afterOperationTwo: MutableMatrix<4, 5, T> = addTwiceRowFourToRowThree.getTransformationMatrix().getMultiplication(fourByFiveSwapped);
const transposedAfterOperationTwo: MutableMatrix<5, 4, T> = afterOperationTwo.getTranspose();
const durationConstructMs = Date.now() - startConstruct;
const startCompute = Date.now();
const multiplication = threeByTwo.getMultiplication(twoByFour);
const withoutLastColumn = multiplication.withoutColumn(3);
const traceOne = LinearAlgebra.trace(withoutLastColumn);
const determinantOne = LinearAlgebra.determinant(withoutLastColumn);

const multiplicationTwo = threeByFour.getMultiplication(fourByFive);

const withoutLastColumnTwo = multiplicationTwo.withoutColumn(4);
const squaredTwo = withoutLastColumnTwo.withoutColumn(3);
const traceTwo = LinearAlgebra.trace(squaredTwo);
const determinantTwo = LinearAlgebra.determinant(squaredTwo);
const durationComputeMs = Date.now() - startCompute;

console.log(multiplication);

console.log("Foud by five: ");
console.log(fourByFive);
console.log("Foud by five swapped: ");
console.log(fourByFiveSwapped);
console.log("After adding twice row four to row three: ");
console.log(afterOperationTwo);

console.log("RRE of last: ");
console.log(LinearAlgebra.reducedRowEchelonForm(afterOperationTwo));

console.log("Modified: ");
console.log(withoutLastColumn);
console.log("Trace: " + traceOne);
console.log("Determinant: " + determinantOne);

console.log(multiplicationTwo);
console.log("Modified (2): ");
console.log(squaredTwo);
console.log("Trace: " + traceTwo);
console.log("Determinant: " + determinantTwo);

console.log("Construct duration ms: " + durationConstructMs);
console.log("Compute duration ms: " + durationComputeMs);