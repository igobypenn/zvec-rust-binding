use std::cmp::Ordering;
use std::collections::HashMap;

use crate::types::MetricType;

pub struct RrfReRanker {
    topn: usize,
    rank_constant: i32,
}

impl RrfReRanker {
    pub fn new(topn: usize) -> Self {
        Self {
            topn,
            rank_constant: 60,
        }
    }

    pub fn with_rank_constant(mut self, rank_constant: i32) -> Self {
        self.rank_constant = rank_constant;
        self
    }

    pub fn topn(&self) -> usize {
        self.topn
    }

    pub fn rank_constant(&self) -> i32 {
        self.rank_constant
    }

    fn rrf_score(&self, rank: usize) -> f64 {
        1.0 / (self.rank_constant as f64 + rank as f64 + 1.0)
    }

    pub fn rerank<K: AsRef<str>>(
        &self,
        query_results: &HashMap<K, Vec<(String, f32)>>,
    ) -> Vec<(String, f32)> {
        let mut rrf_scores: HashMap<String, f64> = HashMap::new();

        for docs in query_results.values() {
            for (rank, (doc_id, _)) in docs.iter().enumerate() {
                let rrf_score = self.rrf_score(rank);
                *rrf_scores.entry(doc_id.clone()).or_insert(0.0) += rrf_score;
            }
        }

        let mut scored: Vec<_> = rrf_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(self.topn);

        scored
            .into_iter()
            .map(|(id, score)| (id, score as f32))
            .collect()
    }
}

pub struct WeightedReRanker {
    topn: usize,
    metric: MetricType,
    weights: HashMap<String, f64>,
}

impl WeightedReRanker {
    pub fn new(topn: usize, metric: MetricType) -> Self {
        Self {
            topn,
            metric,
            weights: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, field: impl Into<String>, weight: f64) -> Self {
        self.weights.insert(field.into(), weight);
        self
    }

    pub fn with_weights(mut self, weights: HashMap<String, f64>) -> Self {
        self.weights = weights;
        self
    }

    pub fn topn(&self) -> usize {
        self.topn
    }

    pub fn metric(&self) -> MetricType {
        self.metric
    }

    fn normalize_score(&self, score: f32) -> f64 {
        match self.metric {
            MetricType::L2 => 1.0 - 2.0 * (score as f64).atan() / std::f64::consts::PI,
            MetricType::Ip => 0.5 + (score as f64).atan() / std::f64::consts::PI,
            MetricType::Cosine => 1.0 - score as f64 / 2.0,
            _ => score as f64,
        }
    }

    pub fn rerank<K: AsRef<str>>(
        &self,
        query_results: &HashMap<K, Vec<(String, f32)>>,
    ) -> Vec<(String, f32)> {
        let mut weighted_scores: HashMap<String, f64> = HashMap::new();

        for (vector_name, docs) in query_results.iter() {
            let weight = self
                .weights
                .get(vector_name.as_ref())
                .copied()
                .unwrap_or(1.0);
            for (doc_id, score) in docs.iter() {
                let normalized = self.normalize_score(*score);
                let weighted = normalized * weight;
                *weighted_scores.entry(doc_id.clone()).or_insert(0.0) += weighted;
            }
        }

        let mut scored: Vec<_> = weighted_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(self.topn);

        scored
            .into_iter()
            .map(|(id, score)| (id, score as f32))
            .collect()
    }
}
