import axios from 'axios';

export interface StatsResponse {
  total_countries: number;
  historical_elections: number;
  upcoming_elections: number;
  overall_accuracy_pct: number;
  overall_correct: number;
  clean_accuracy_pct: number;
  clean_correct: number;
  clean_total: number;
  cv_mean_cb: number;
  cv_mean_xgb: number;
  cv_mean_ensemble: number;
}

export interface UpcomingElection {
  election_id: string;
  source: string;
  country_code: string;
  country_name: string;
  region: string;
  year: number;
  election_type: 'Executive' | 'Legislative';
  is_scheduled: number;
  clean_index: number | null;
  gdp_growth: number | null;
  raw_probability: number;
  predicted_outcome: number;
  predicted_winner: 'Incumbent' | 'Challenger';
  raw_confidence: number;
  adjusted_confidence: number;
  data_completeness: number;
  data_source_flags: number;
  target_outcome: number | null;
  election_date: string | null;
}

export interface HistoricalElection {
  election_id: string;
  source: string;
  country_code: string;
  country_name: string;
  region: string;
  year: number;
  election_type: 'Executive' | 'Legislative';
  is_scheduled: number;
  clean_index: number | null;
  raw_probability: number;
  predicted_outcome: number;
  predicted_winner: 'Incumbent' | 'Challenger';
  raw_confidence: number;
  adjusted_confidence: number;
  data_completeness: number;
  data_source_flags: number;
  target_outcome: number;
  target_outcome_int: number;
  predicted_outcome_int: number;
  is_correct: number;
  election_date: string | null;
}

export interface CountryAccuracy {
  rank: number;
  country_code: string;
  country_name: string;
  region: string;
  total: number;
  correct: number;
  accuracy_pct: number;
  exec_total: number;
  exec_correct: number;
  leg_total: number;
  leg_correct: number;
}

export interface CalibrationBin {
  bin: string;
  midpoint: number;
  exec_count: number;
  leg_count: number;
  exec_accuracy: number | null;
  leg_accuracy: number | null;
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export interface DiagnosticsResponse {
  cv_folds: Array<{
    Origin: string;
    'CatBoost Classifier': number;
    'XGBoost Classifier': number;
    'Voting Ensemble': number;
  }>;
  mean_catboost: number;
  mean_xgboost: number;
  mean_ensemble: number;
  country_accuracy_distribution: Array<{
    bin: string;
    count: number;
  }>;
  calibration_bins: CalibrationBin[];
  confusion_matrix_exec: ConfusionMatrix;
  confusion_matrix_leg: ConfusionMatrix;
}

export interface ShapFeature {
  feature: string;
  human_name: string;
  literature_info: string;
  value: number | null;
  contribution: number;
  direction: 'Favors Incumbent' | 'Favors Challenger';
}

export interface ShapResponse {
  election_id: string;
  country_code: string;
  year: number;
  election_type: string;
  predicted_winner: string;
  predicted_probability: number;
  actual_outcome: string | null;
  is_clean: boolean;
  top_features: ShapFeature[];
}

export interface VdemMapRecord {
  country_code: string;
  country_name: string;
  latitude: number | null;
  longitude: number | null;
  clean_elections_index: number | null;
  polyarchy_index: number | null;
  regime_type: number | null;
  regime_name: string;
}

export interface FeatureImportanceRecord {
  feature: string;
  human_name: string;
  importance: number;
  explanation: string;
}

// In-memory static cache for fast lookup
let shapMapCache: Record<string, ShapResponse> | null = null;
let vdemMapCache: Record<number, VdemMapRecord[]> | null = null;

export const fetchStats = async (): Promise<StatsResponse> => {
  const res = await axios.get<StatsResponse>('/data/stats.json');
  return res.data;
};

export const fetchUpcoming = async (): Promise<UpcomingElection[]> => {
  const res = await axios.get<UpcomingElection[]>('/data/upcoming.json');
  return res.data;
};

export const fetchHistorical = async (): Promise<HistoricalElection[]> => {
  const res = await axios.get<HistoricalElection[]>('/data/historical.json');
  return res.data;
};

export const fetchCountryAccuracy = async (): Promise<CountryAccuracy[]> => {
  const res = await axios.get<CountryAccuracy[]>('/data/country_accuracy.json');
  return res.data;
};

export const fetchDiagnostics = async (): Promise<DiagnosticsResponse> => {
  const res = await axios.get<DiagnosticsResponse>('/data/model_diagnostics.json');
  return res.data;
};

export const fetchShapExplanation = async (electionId: string): Promise<ShapResponse> => {
  if (!shapMapCache) {
    const res = await axios.get<Record<string, ShapResponse>>('/data/shap_explanations.json');
    shapMapCache = res.data;
  }
  const match = shapMapCache[electionId];
  if (!match) {
    throw new Error(`SHAP explanation for election ${electionId} not found`);
  }
  return match;
};

export const fetchVdemMap = async (year: number): Promise<VdemMapRecord[]> => {
  if (!vdemMapCache) {
    const res = await axios.get<Record<number, VdemMapRecord[]>>('/data/vdem_map.json');
    vdemMapCache = res.data;
  }
  return vdemMapCache[year] || [];
};

export const fetchFeatureImportances = async (): Promise<FeatureImportanceRecord[]> => {
  const res = await axios.get<FeatureImportanceRecord[]>('/data/feature_importances.json');
  return res.data;
};
