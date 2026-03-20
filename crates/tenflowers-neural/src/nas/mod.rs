//! Neural Architecture Search (NAS) — unified module.
//!
//! Re-exports from both the legacy algorithms (DARTS-cell mixed-ops, aging evolution,
//! Lottery Ticket) and the new comprehensive NAS suite (DARTS optimizer, Gumbel-softmax,
//! evolutionary search with crossover, one-shot supernet, and architecture serialization).

pub mod extensions;
pub mod legacy;

// ── Legacy exports ────────────────────────────────────────────────────────────
pub use legacy::{
    AgingEvolutionNas, ArchitectureEvaluator, DartsCell, LotteryTicketPruner, MixedOp, OpsChoice,
    RandomSearchNas, SearchSpace, TicketState,
};

// ── New comprehensive NAS exports ─────────────────────────────────────────────
pub use extensions::{
    decode_architecture_string,
    // Serialization
    encode_architecture_string,
    network_stats,

    CellConfig,
    CellEdge,
    CellEncoding,
    // DARTS
    DartsConfig,
    DartsOptimizer,

    DartsState,
    // Evolutionary NAS (new, richer version)
    EvoNasConfig,
    EvolutionResult,

    EvolutionaryNas,
    // Gumbel-Softmax
    GumbelSoftmax,

    // Logger
    NasLogger,
    NasSummary,

    NetworkEncoding,
    NetworkStats,
    NodeConfig,
    // One-shot NAS
    OneShotConfig,
    OneShotNas,

    // Architecture encoding
    OpType,
    // Random NAS
    RandomNasConfig,
    RandomNasSearch,
};
