/**
 * 2-D semantic map of a paper collection.
 *
 * Every chunk of every paper (plus one centroid per paper) is placed on a
 * plane so that nearby points talk about similar things.
 *
 * UMAP (McInnes et al., 2018) builds a k-nearest-neighbour graph in the
 * original space and lays it out in 2-D so that neighbourhoods are preserved;
 * it is the standard choice for embedding maps (BERTopic uses it too). It is
 * stochastic, so we feed it a seeded PRNG and the same input always gives the
 * same picture.
 *
 * With very few points UMAP's neighbourhood graph is meaningless, and without
 * embeddings there are no dense vectors to feed it, so we fall back to
 * classical (Torgerson) MDS: double-centre the squared distance matrix and
 * take its two leading eigenvectors, found by power iteration with deflation.
 * MDS reproduces the pairwise distances as faithfully as a plane allows.
 */
import { UMAP } from "umap-js";
import type { SurveyResult } from "@/types";
import { similarityMatrix } from "@/nlp/lexrank";
import { toDistance } from "./clustering";

export type SurveyMap = SurveyResult["map"];

export interface MapItem {
  paperId: string;
  chunkId?: string;
  isCentroid: boolean;
}

/** Below this many points UMAP's k-NN graph is too sparse to be worth more than MDS. */
const MIN_UMAP_POINTS = 8;
const UMAP_NEIGHBOURS = 15;
const DEFAULT_SEED = 42;

/** mulberry32: a tiny, fast, well-distributed 32-bit PRNG; same seed, same sequence. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Projects dense vectors with UMAP when there are enough of them, otherwise
 * (or if UMAP fails) with classical MDS on cosine distances.
 */
export function projectDense(items: MapItem[], vectors: Float32Array[], seed = DEFAULT_SEED): SurveyMap {
  if (vectors.length >= MIN_UMAP_POINTS) {
    try {
      const coordinates = new UMAP({
        nComponents: 2,
        nNeighbors: Math.min(UMAP_NEIGHBOURS, vectors.length - 1),
        minDist: 0.1,
        random: mulberry32(seed),
      }).fit(vectors.map((v) => Array.from(v)));
      if (coordinates.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
        return { points: toPoints(items, coordinates), method: "umap" };
      }
    } catch {
      // Degenerate inputs (e.g. identical vectors) can break the k-NN graph; MDS still works.
    }
  }
  return projectSimilarity(items, similarityMatrix(vectors), seed);
}

/** Classical MDS from a similarity matrix (used for TF-IDF vectors and small inputs). */
export function projectSimilarity(items: MapItem[], similarity: number[][], seed = DEFAULT_SEED): SurveyMap {
  return { points: toPoints(items, classicalMds(toDistance(similarity), seed)), method: "mds" };
}

function toPoints(items: MapItem[], coordinates: number[][]): SurveyMap["points"] {
  const scaled = rescale(coordinates);
  return items.map((item, i) => ({
    x: scaled[i][0],
    y: scaled[i][1],
    paperId: item.paperId,
    ...(item.chunkId ? { chunkId: item.chunkId } : {}),
    isCentroid: item.isCentroid,
  }));
}

/** Centres the layout and scales it into [-1, 1] so the UI does not depend on the method's units. */
function rescale(coordinates: number[][]): number[][] {
  if (coordinates.length === 0) return [];
  const mean = [0, 1].map((d) => coordinates.reduce((s, p) => s + p[d], 0) / coordinates.length);
  const centred = coordinates.map((p) => [p[0] - mean[0], p[1] - mean[1]]);
  const extent = Math.max(...centred.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
  return extent > 0 ? centred.map((p) => [p[0] / extent, p[1] / extent]) : centred;
}

// ─────────────────────────────── Classical MDS ──────────────────────────────

/** Torgerson MDS: coordinates = top-2 eigenvectors of B = -1/2 J D² J, scaled by sqrt(eigenvalue). */
export function classicalMds(distance: number[][], seed = DEFAULT_SEED): number[][] {
  const n = distance.length;
  if (n === 0) return [];
  if (n === 1) return [[0, 0]];
  const b = doubleCentre(distance);
  const random = mulberry32(seed);
  const axes: { value: number; vector: number[] }[] = [];
  for (let component = 0; component < 2; component++) {
    const axis = leadingEigen(b, random);
    axes.push(axis);
    deflate(b, axis.value, axis.vector);
  }
  return Array.from({ length: n }, (_, i) => axes.map(({ value, vector }) => vector[i] * Math.sqrt(Math.max(0, value))));
}

function doubleCentre(distance: number[][]): number[][] {
  const n = distance.length;
  const squared = distance.map((row) => row.map((d) => d * d));
  const rowMean = squared.map((row) => row.reduce((s, v) => s + v, 0) / n);
  const grandMean = rowMean.reduce((s, v) => s + v, 0) / n;
  return squared.map((row, i) => row.map((v, j) => -0.5 * (v - rowMean[i] - rowMean[j] + grandMean)));
}

const POWER_ITERATIONS = 300;
const POWER_TOLERANCE = 1e-10;

/** Dominant eigenpair of a symmetric matrix by power iteration from a seeded random start. */
function leadingEigen(matrix: number[][], random: () => number): { value: number; vector: number[] } {
  const n = matrix.length;
  let vector = unit(Array.from({ length: n }, () => random() - 0.5));
  let value = 0;
  for (let step = 0; step < POWER_ITERATIONS; step++) {
    const next = multiply(matrix, vector);
    const length = Math.hypot(...next);
    if (length === 0) return { value: 0, vector };
    const normalised = next.map((v) => v / length);
    const change = normalised.reduce((s, v, i) => s + Math.abs(v - vector[i]), 0);
    vector = normalised;
    value = length;
    if (change < POWER_TOLERANCE) break;
  }
  // Rayleigh quotient gives the signed eigenvalue (power iteration only sees its magnitude).
  value = dotArrays(vector, multiply(matrix, vector));
  return { value, vector };
}

function deflate(matrix: number[][], value: number, vector: number[]): void {
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix.length; j++) matrix[i][j] -= value * vector[i] * vector[j];
  }
}

function multiply(matrix: number[][], vector: number[]): number[] {
  return matrix.map((row) => dotArrays(row, vector));
}

function dotArrays(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function unit(vector: number[]): number[] {
  const length = Math.hypot(...vector);
  return length > 0 ? vector.map((v) => v / length) : vector;
}
