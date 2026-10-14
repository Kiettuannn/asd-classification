/**
 * Domain types for cohort phenotypic data and model benchmarking.
 *
 * These types represent ABIDE subject metadata and model performance records.
 * Implementation of cohort analytics rendering is deferred.
 */

/** Diagnostic label for a cohort subject. */
export type DiagnosisLabel = 'ASD' | 'TD';

/** Sex label as encoded in ABIDE phenotypic data. */
export type SexLabel = 'M' | 'F';

/** Acquisition site from the ABIDE dataset. */
export interface AcquisitionSite {
  site_id: string;
  site_name: string;
  num_subjects: number;
}

/** Phenotypic metadata for a single ABIDE subject. */
export interface SubjectPhenotype {
  subject_id: string;
  diagnosis: DiagnosisLabel;
  site: string;
  age: number;
  sex: SexLabel;
  /** Full-scale IQ score */
  iq?: number;
  /** Autism Diagnostic Observation Schedule total score */
  ados_total?: number;
}

/** Summary statistics for a cohort subset. */
export interface CohortSummary {
  total_subjects: number;
  asd_count: number;
  td_count: number;
  mean_age: number;
  sites: AcquisitionSite[];
}

/** A single model benchmark entry from the leaderboard. */
export interface ModelBenchmark {
  model_name: string;
  accuracy: number;
  sensitivity: number;
  specificity: number;
  auc_roc: number;
  /** ISO-8601 datetime string */
  evaluated_at: string;
}
