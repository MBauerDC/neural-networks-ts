import {Readable} from 'stream';
import * as fs from 'fs';
import { Column, Dimension, Matrix, MatrixContent, Row } from './../../node_modules/matrices/src/matrix';
import { GenericMutableColumn, GenericMutableMatrix, GenericMutableRow, MutableColumn, MutableMatrix, MutableRow } from './../../node_modules/matrices/src/mutable';
import { LabelledDataPoint } from '../network/network';

function createFileReadStream(location: string, encoding: BufferEncoding | undefined = 'utf-8'): Readable {
  return fs.createReadStream(location, {encoding});
}

async function* chunksToLines(chunkIterable: Iterable<string> | AsyncIterable<string>): AsyncIterable<string> {
  let previous = '';
  for await (const chunk of chunkIterable) {
    let startSearch = previous.length;
    previous += chunk;
    while (true) {
      const eolIndex = previous.indexOf('\n', startSearch);
      if (eolIndex < 0) break;
      // line includes the EOL
      const line = previous.slice(0, eolIndex+1);
      yield line;
      previous = previous.slice(eolIndex+1);
      startSearch = 0;
    }
  }
  if (previous.length > 0) {
    yield previous;
  }
}

type CSVParsedContentTypeName = 'null' | 'string' | 'boolean' | 'int' | 'float';
type CSVParsedContentType = null | string | boolean | number;

type ArrayToRowConverter = <M extends Dimension, O extends MatrixContent>(array: Array<any>, transformer: (i:any) => O, fromIdx: number, columns: Dimension) => Row<M, O>;
type ArrayToColumnConverter = <N extends Dimension, O extends MatrixContent>(array: Array<any>, transformer: (i:any) => O, fromIdx: number, columns: Dimension) => Column<N, O>;

type NestedArrayToMatrixConverter = <N extends Dimension, M extends Dimension, I extends any, O extends MatrixContent>(array: Array<Array<I>>, transformer: (i:I) => O, rows: Dimension, columns: Dimension) => Matrix<N, M, O>;

type CSVLineToRowsConverter = <I extends CSVParsedContentType, O extends MatrixContent>(line: Array<I>, rowSpecs: Array<Dimension>, transformer: (i:I) => O) => Row<Dimension, O>[];
type CSVLineToColumnsConverter = <I extends CSVParsedContentType, O extends MatrixContent>(line: Array<I>, columnSpecs: Array<Dimension>, transformer: (i:I) => O) => Column<Dimension, O>[];

type CSVLinesToRowMatricesConverter = <I extends CSVParsedContentType, O extends MatrixContent>(lines: Array<Array<I>>, rowSpecs: Array<Dimension>, transformer: (i:I) => O) => Matrix<Dimension, Dimension, O>[];
type CSVLinesToColumnMatricesConverter = <I extends CSVParsedContentType, O extends MatrixContent>(lines: Array<Array<I>>, columnSpecs: Array<Dimension>, transformer: (i:I) => O) => Matrix<Dimension, Dimension, O>[];

const csvArrayToRowConverter: ArrayToRowConverter = <M extends Dimension, I extends CSVParsedContentType, O extends MatrixContent>(array: Array<I>, transformer: (i:I) => O, fromIdx: number, columns: Dimension) => {
    const mappedArray =  array.map(transformer).slice(fromIdx, columns);
    return new GenericMutableRow(mappedArray, columns as M);
}

const csvArrayToColumnConverter: ArrayToColumnConverter = <N extends Dimension, I extends CSVParsedContentType, O extends MatrixContent>(array: Array<I>, transformer: (i:I) => O, fromIdx: number, columns: Dimension) => {
    const mappedArray =  array.map(transformer).slice(fromIdx, columns);
    return new GenericMutableColumn(mappedArray, columns as N);
}

const csvLineArrayToRowsConverter: CSVLineToRowsConverter = <I extends CSVParsedContentType, O extends MatrixContent>(line: Array<I>, rowSpecs: Array<Dimension>, transformer: (i:I) => O): Row<Dimension, O>[] => {
    let fromIdx = 0;
    return rowSpecs.map((columns) => {
        const row = csvArrayToRowConverter(line, transformer, fromIdx, columns);
        fromIdx += columns;
        return row;
    });
}

const csvLineArrayToColumnsConverter: CSVLineToColumnsConverter = <I extends CSVParsedContentType, O extends MatrixContent>(line: Array<I>, columnSpecs: Array<Dimension>, transformer: (i:I) => O): Column<Dimension, O>[] => {
    let fromIdx = 0;
    return columnSpecs.map((rows) => {
        const column = csvArrayToColumnConverter(line, transformer, fromIdx, rows);
        fromIdx += rows;
        return column;
    });
}

const csvLineArraysToRowMatricesConverter: CSVLinesToRowMatricesConverter = <I extends CSVParsedContentType, O extends MatrixContent>(lines: Array<Array<I>>, rowSpecs: Array<Dimension>, transformer: (i:I) => O): Matrix<Dimension, Dimension, O>[] => {
    const rowsArray = lines.map((line) => csvLineArrayToRowsConverter(line, rowSpecs, transformer));
    const matricesArray: MutableMatrix<Dimension, Dimension, O>[] = [];
    for (let i = 0; i < rowSpecs.length; i++) {
        const rows = rowsArray.map((rowArray) => rowArray[i]);
        const matrix = new GenericMutableMatrix(null, rows as MutableRow<Dimension, O>[], null, rowsArray.length, rowSpecs[i]);
        matricesArray.push(matrix);
    }
    return matricesArray
}

const csvLineArraysToColumnMatricesConverter: CSVLinesToColumnMatricesConverter = <I extends CSVParsedContentType, O extends MatrixContent>(lines: Array<Array<I>>, columnSpecs: Array<Dimension>, transformer: (i:I) => O): Matrix<Dimension, Dimension, O>[] => {
    const columnsArray = lines.map((line) => csvLineArrayToColumnsConverter(line, columnSpecs, transformer));
    const matricesArray: MutableMatrix<Dimension, Dimension, O>[] = [];
    for (let i = 0; i < columnSpecs.length; i++) {
        const columns = columnsArray.map((columnsArray) => columnsArray[i]);
        const matrix = new GenericMutableMatrix(null, null, columns as MutableColumn<Dimension, O>[], columnSpecs[i], columnsArray.length);
        matricesArray.push(matrix);
    }
    return matricesArray
}

async function *generateCSVLines(location: string, permittedTypes: Array<CSVParsedContentTypeName>, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<Array<CSVParsedContentType>> {
    for await (const line of chunksToLines(createFileReadStream(location, encoding))) {
        yield parseCSVLine(line, permittedTypes);
    }
}

async function *generateCSVRowMatrices<O extends MatrixContent>(rowSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, linesIterator: Iterable<Array<CSVParsedContentType>> | AsyncIterable<Array<CSVParsedContentType>>, rowsPerMatrix: number = 1): AsyncIterable<Matrix<Dimension, Dimension, MatrixContent>[]> {
    for await (const lines of collect(linesIterator, rowsPerMatrix)) {
        yield csvLineArraysToRowMatricesConverter(lines, rowSpecs, transformer); // Rewrite 
    }
}

async function *generateCSVColumnMatrices<O extends MatrixContent>(columnSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, linesIterator: Iterable<Array<CSVParsedContentType>> | AsyncIterable<Array<CSVParsedContentType>>, columnsPerMatrix: number = 1): AsyncIterable<Matrix<Dimension, Dimension, MatrixContent>[]> {
    for await (const lines of collect(linesIterator, columnsPerMatrix)) {
        yield csvLineArraysToColumnMatricesConverter(lines, columnSpecs, transformer); // Rewrite 
    }
}

async function *readCSVToRows<O extends MatrixContent>(location: string, rowSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, permittedTypes: Array<CSVParsedContentTypeName>, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<Row<Dimension, MatrixContent>[]> {
    const linesIterator = generateCSVLines(location, permittedTypes, encoding);
    for await (const line of linesIterator) {
        yield csvLineArrayToRowsConverter(line, rowSpecs, transformer);
    }
}

async function *readCSVToColumns<O extends MatrixContent>(location: string, columnSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, permittedTypes: Array<CSVParsedContentTypeName>, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<Column<Dimension, MatrixContent>[]> {
    const linesIterator = generateCSVLines(location, permittedTypes, encoding);
    for await (const line of linesIterator) {
        yield csvLineArrayToColumnsConverter(line, columnSpecs, transformer);
    }
}

async function *readCSVToRowMatrices<O extends MatrixContent>(location: string, rowSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, permittedTypes: Array<CSVParsedContentTypeName>, rowsPerMatrix: number = 1, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<Matrix<Dimension, Dimension, MatrixContent>[]> {
    const linesIterator = generateCSVLines(location, permittedTypes, encoding);
    yield* generateCSVRowMatrices(rowSpecs, transformer, linesIterator, rowsPerMatrix);
}

async function *readCSVToColumnMatrices<O extends MatrixContent>(location: string, columnSpecs: Array<Dimension>, transformer: (i: CSVParsedContentType) => O, permittedTypes: Array<CSVParsedContentTypeName>, columnsPerMatrix: number = 1, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<Matrix<Dimension, Dimension, MatrixContent>[]> {
    const linesIterator = generateCSVLines(location, permittedTypes, encoding);
    yield* generateCSVColumnMatrices(columnSpecs, transformer, linesIterator, columnsPerMatrix);
}

export async function *readCSVToLabelledDataPoints<I extends Dimension, O extends Dimension>(location: string, columnSpecs: [Dimension, Dimension], transformer: (i: CSVParsedContentType) => number, inputColumnIdx: 0 | 1, encoding: BufferEncoding | undefined = 'utf-8'): AsyncIterable<LabelledDataPoint<I, O>> {
    for await (const ioColumns of readCSVToColumns(location, columnSpecs, transformer, ['int', 'float'], encoding)) {
        const outputColumnIdx = inputColumnIdx === 0 ? 1 : 0;
        const input = ioColumns[inputColumnIdx] as Column<I, number>;
        const output = ioColumns[outputColumnIdx] as Column<O, number>;
        yield {input: input, output: output};
    }
}

function parseCSVLine(line: string, permittedTypes: Array<CSVParsedContentTypeName> = ['null', 'string', 'boolean', 'int', 'float']): Array<any> {
    const permitsNull = permittedTypes.includes('null');
    const permitsString = permittedTypes.includes('string');
    const permitsBoolean = permittedTypes.includes('boolean');
    const permitsInt = permittedTypes.includes('int');
    const permitsFloat = permittedTypes.includes('float');

    return line.match(/\s*(\"[^"]*\"|'[^']*'|[^,]*)\s*(,|$)/g)?.map(function (line) {
      let m;
      if (permitsNull) {
        if (m = line.match(/^\s*,?$/)) return null; // null value
      }
      if (permitsString) {
        if (m = line.match(/^\s*\"([^"]*)\"\s*,?$/)) return m[1]; // Double Quoted Text
        if (m = line.match(/^\s*'([^']*)'\s*,?$/)) return m[1]; // Single Quoted Text
      }
      if (permitsBoolean) {
        if (m = line.match(/^\s*(true|false)\s*,?$/)) return m[1] === "true"; // Boolean
      }
      if (permitsInt) {
        if (m = line.match(/^\s*((?:\+|\-)?\d+)\s*,?$/)) return parseInt(m[1]); // Integer Number
      }
      if (permitsFloat) {
        if (m = line.match(/^\s*((?:\+|\-)?\d*\.\d*)\s*,?$/)) return parseFloat(m[1]); // Floating Number
      }
      if (permitsString) {
        if (m = line.match(/^\s*(.*?)\s*,?$/)) return m[1]; // Unquoted Text
      }
      return line;
    }) ?? [];
  }

  async function *collect<T>(iterable: Iterable<T> | AsyncIterable<T>, size: number): AsyncIterable<T[]> {
    let buffer: Array<T> = [];
    for await (const item of iterable) {
      buffer.push(item);
      if (buffer.length >= size) {
        yield buffer;
        buffer = [];
      }
    }
  }