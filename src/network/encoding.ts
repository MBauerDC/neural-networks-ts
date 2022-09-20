import { Column, Dimension, MatrixContent } from "../matrices/matrix";
import { GenericSparseMutableColumn } from "../matrices/sparse";

type EncodingSchemeType = "one-hot" | "binary" | "ordinal";
type EncodingScheme<I extends any, O extends MatrixContent, N extends Dimension> = (i: I) => Column<N, O>;
type DecodingSchemeType = EncodingSchemeType;
type DecodingScheme<N extends Dimension, I extends MatrixContent, O extends any> = (i: Column<N, I>) => O;
type EncodingDecodingScheme<N extends Dimension, A extends any, B extends MatrixContent> = { encoding: EncodingScheme<A, B, N>, decoding: DecodingScheme<N, B, A> };
type LabelType = string|number;
type LabelLookup<T extends LabelType> = (idx: number) => T;

class GenericEncodingDecodingScheme<N extends Dimension, A extends any, B extends MatrixContent> implements EncodingDecodingScheme<N, A, B> {
    constructor(public readonly encoding: EncodingScheme<A, B, N>, public readonly decoding: DecodingScheme<N, B, A>){};
}

const oneHotEncodingSchemeCtor: <I extends any, N extends Dimension>(oneOf: N, lookupFn: (i:I) => number) => EncodingScheme<I, number, N> = <I extends any, N extends Dimension>(oneOf: N, lookupFn: (i:I) => number) => (i:I) => {
    const idxToSet = lookupFn(i);
    const sparseData: Record<number, number> = {};
    sparseData[idxToSet] = 1;
    return new GenericSparseMutableColumn(sparseData, oneOf);
}

const oneHotDecodingSchemeCtor: <N extends Dimension, T extends LabelType>(labelLookupFn: LabelLookup<T>) => DecodingScheme<N, number, T> = <N extends Dimension, T extends LabelType>(labelLookupFn: LabelLookup<T>) => (i: Column<N, number>) => {
    let idx = 0;
    for (let rowValue of i.generateColumn(0)) {
        if (rowValue === 1) {
            return labelLookupFn(idx);
        }
        idx++;
    }
}

const createOneHotScheme = <I extends LabelType, N extends Dimension>(oneOf: N, lookupFn: (i:I) => number, labelLookupFn: (idx: number) => I) => {
    return new GenericEncodingDecodingScheme<N, I, number>(oneHotEncodingSchemeCtor(oneOf, lookupFn), oneHotDecodingSchemeCtor(labelLookupFn));
}

export { EncodingSchemeType, EncodingScheme, DecodingSchemeType, DecodingScheme, EncodingDecodingScheme, LabelType, LabelLookup, GenericEncodingDecodingScheme, createOneHotScheme };

