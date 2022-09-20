import { Dimension, Matrix } from "../matrices/matrix";
import { ActivationFunctionClass } from "./activation";

export class LayerOptions<N extends Dimension, F extends number> {
    constructor(
        public activation: ActivationFunctionClass,
        public size: N,
        public initialValues: Matrix<N, 1, F>,
        public initialBiases: Matrix<N, 1, F>
    ) {}

    withInitialValues(newValues: Matrix<N, 1, F>): LayerOptions<N, F> {
        return new LayerOptions(this.activation, this.size, newValues, this.initialBiases);
    }
    withInitialBiases(newBiases: Matrix<N, 1, F>): LayerOptions<N, F> {
        return new LayerOptions(this.activation, this.size, this.initialValues, newBiases);
    }
}

export function createLayerOptions<N extends Dimension, F extends number>(
    activation: ActivationFunctionClass,
    size: N,
    initialValues: Matrix<N, 1, F>,
    initialBiases: Matrix<N, 1, F>
): LayerOptions<N, F> {
    return new LayerOptions(activation, size, initialValues, initialBiases);
}

export interface Layer<N extends Dimension, F extends number, O extends LayerOptions<N, F>> {
    readonly options: O;
    withOptions<M extends Dimension, G extends number>(newOptions: LayerOptions<M, G>): Layer<M, G, LayerOptions<M, G>>;
    values: Matrix<N, 1, F>;
    biases: Matrix<N, 1, F>;
    reset(): void;
}

export class LayerImpl<N extends Dimension, F extends number, O extends LayerOptions<N, F>> implements Layer<N, F, O> {
    constructor(public readonly options: O, public values: Matrix<N, 1, F>, public biases: Matrix<N, 1, F>) {
        this.values = options.initialValues;
        this.biases = options.initialBiases;
    }

    public withOptions<M extends Dimension, G extends number>(newOptions: LayerOptions<M, G>): Layer<M, G, LayerOptions<M, G>> {
        return new LayerImpl(newOptions, newOptions.initialValues, newOptions.initialBiases);
    }

    public reset(): void {
        this.values = this.options.initialValues;
        this.biases = this.options.initialBiases;
    }
}

export function createLayer<N extends Dimension, F extends number>(options: LayerOptions<N, F>): Layer<N, F, LayerOptions<N, F>> {	
    return new LayerImpl(options, options.initialValues, options.initialBiases);
}

