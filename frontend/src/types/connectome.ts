/**
 * Domain types for brain connectome visualisation.
 *
 * These types represent the data structures used when rendering FC matrices,
 * ROI graphs and saliency overlays. Visualization implementation is deferred;
 * these interfaces freeze the architectural contract.
 */

/** A 2-D functional connectivity correlation matrix (N×N). */
export interface FCMatrix {
  /** Atlas-specific ROI count (e.g. 200 for CC200). */
  num_rois: number;
  /** Flat row-major representation of the N×N matrix values. */
  values: number[];
}

/** A labelled region-of-interest node in the atlas. */
export interface ROINode {
  /** 0-based ROI index */
  index: number;
  /** Human-readable label from the atlas (e.g. "Frontal_Inf_Orb_L") */
  label: string;
  /** MNI-space centroid coordinates [x, y, z] in mm */
  mni_coords?: [number, number, number];
  /** Canonical network membership (e.g. "Default Mode", "Salience") */
  network?: string;
}

/** A weighted undirected edge between two ROIs. */
export interface ROIEdge {
  source: number;
  target: number;
  weight: number;
}

/** A complete graph representation of ROI connectivity. */
export interface ConnectomeGraph {
  nodes: ROINode[];
  edges: ROIEdge[];
}
