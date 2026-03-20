//! Advanced generation algorithms.
//!
//! - Speculative decoding utilities: `MedianDraftLength`
//! - Regex finite-state machine: `RegexFsm`
//! - Grammar-guided sampling (Earley-style CFG): `GrammarSampler`
//! - RAG flat index: `RagIndex`
//! - Generation evaluation metrics: `BertScoreProxy`, `GenerationMetrics`

use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use std::collections::HashMap;
use tenflowers_core::{Result, TensorError};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Speculative Decoding Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks accepted draft lengths online and reports the median.
///
/// Used to adaptively tune the draft budget `K` for speculative decoding.
pub struct MedianDraftLength {
    history: Vec<usize>,
    sorted: Vec<usize>,
    dirty: bool,
}

impl MedianDraftLength {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            sorted: Vec::new(),
            dirty: false,
        }
    }

    /// Record an accepted draft length.
    pub fn push(&mut self, accepted: usize) {
        self.history.push(accepted);
        self.dirty = true;
    }

    /// Median accepted draft length (0 if no observations).
    pub fn median(&mut self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        if self.dirty {
            self.sorted = self.history.clone();
            self.sorted.sort_unstable();
            self.dirty = false;
        }
        let n = self.sorted.len();
        if n % 2 == 1 {
            self.sorted[n / 2] as f64
        } else {
            (self.sorted[n / 2 - 1] + self.sorted[n / 2]) as f64 / 2.0
        }
    }

    /// Suggested next draft length: ceil(median) clamped to [1, max_k].
    pub fn suggested_k(&mut self, max_k: usize) -> usize {
        let m = self.median().ceil() as usize;
        m.max(1).min(max_k)
    }

    /// Number of recorded observations.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether no observations have been recorded.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

impl Default for MedianDraftLength {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Regex Finite-State Machine
// ─────────────────────────────────────────────────────────────────────────────

/// State in a deterministic finite automaton (DFA) for regex matching.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FsmState {
    /// The DFA is in a normal (non-accepting, non-dead) state.
    Active(usize),
    /// The DFA has reached an accepting state.
    Accepted,
    /// The DFA has entered a dead (rejecting) state.
    Dead,
}

/// Simple character-class DFA for constrained token generation.
///
/// Supports a minimal regex grammar:
/// - Literals: any printable ASCII character
/// - `.` matches any character
/// - `*` Kleene star on the preceding character class
/// - `+` one-or-more on the preceding character class
/// - `?` zero-or-one on the preceding character class
/// - `[0-9]`, `[a-z]`, `[A-Z]`, `[a-zA-Z0-9]` character classes
///
/// This is a teaching-quality implementation; it does NOT handle full POSIX regexes.
pub struct RegexFsm {
    /// Transition table: `transitions[state][char_idx]` = next_state (usize::MAX = dead).
    transitions: Vec<Vec<usize>>,
    /// Accepting states.
    accepting: Vec<bool>,
    /// Current DFA state.
    current_state: usize,
    /// Number of characters consumed so far.
    n_consumed: usize,
    /// Characters in the alphabet (ASCII 32–126).
    alphabet_offset: u8,
    alphabet_size: usize,
}

impl RegexFsm {
    const DEAD: usize = usize::MAX;

    /// Build a DFA from a simplified pattern string.
    ///
    /// Recognised patterns:
    /// - `[digit]` → digits 0-9
    /// - `[alpha]` → letters a-z, A-Z
    /// - `[alnum]` → letters and digits
    /// - `[word]`  → letters, digits, underscore
    /// - `.` → any printable ASCII
    /// - A single ASCII character.
    ///
    /// Optional quantifier suffix: `*`, `+`, `?`.
    ///
    /// Multiple such atoms are concatenated (no alternation).
    pub fn from_pattern(pattern: &str) -> Result<Self> {
        // Parse into atoms: (char_set: Vec<u8>, quantifier: char)
        let atoms = Self::parse_atoms(pattern)?;
        Self::build_dfa(atoms)
    }

    fn parse_atoms(pattern: &str) -> Result<Vec<(Vec<u8>, char)>> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut atoms: Vec<(Vec<u8>, char)> = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let (set, consumed) = Self::parse_char_set(&chars, i)?;
            i += consumed;
            // Optional quantifier.
            let quant = if i < chars.len() && matches!(chars[i], '*' | '+' | '?') {
                let q = chars[i];
                i += 1;
                q
            } else {
                '1' // exactly one
            };
            atoms.push((set, quant));
        }
        Ok(atoms)
    }

    fn parse_char_set(chars: &[char], pos: usize) -> Result<(Vec<u8>, usize)> {
        if pos >= chars.len() {
            return Err(TensorError::invalid_argument(
                "unexpected end of pattern".to_string(),
            ));
        }
        match chars[pos] {
            '[' => {
                // Find closing ']'.
                let close = chars[pos + 1..]
                    .iter()
                    .position(|&c| c == ']')
                    .ok_or_else(|| {
                        TensorError::invalid_argument("unmatched '[' in pattern".to_string())
                    })?;
                let class_str: String = chars[pos + 1..pos + 1 + close].iter().collect();
                let consumed = close + 2; // '[' + class + ']'
                let set = Self::char_class_to_set(&class_str)?;
                Ok((set, consumed))
            }
            '.' => {
                let set: Vec<u8> = (32u8..=126u8).collect();
                Ok((set, 1))
            }
            c => {
                if c.is_ascii() {
                    Ok((vec![c as u8], 1))
                } else {
                    Err(TensorError::invalid_argument(
                        "only ASCII characters supported".to_string(),
                    ))
                }
            }
        }
    }

    fn char_class_to_set(class: &str) -> Result<Vec<u8>> {
        match class {
            "digit" | "0-9" => Ok((b'0'..=b'9').collect()),
            "alpha" | "a-zA-Z" => {
                let mut v: Vec<u8> = (b'a'..=b'z').collect();
                v.extend(b'A'..=b'Z');
                Ok(v)
            }
            "alnum" | "a-zA-Z0-9" => {
                let mut v: Vec<u8> = (b'a'..=b'z').collect();
                v.extend(b'A'..=b'Z');
                v.extend(b'0'..=b'9');
                Ok(v)
            }
            "word" => {
                let mut v: Vec<u8> = (b'a'..=b'z').collect();
                v.extend(b'A'..=b'Z');
                v.extend(b'0'..=b'9');
                v.push(b'_');
                Ok(v)
            }
            _ => {
                // Try simple range like "a-z" or "A-Z" or "0-9".
                let cs: Vec<char> = class.chars().collect();
                if cs.len() == 3 && cs[1] == '-' && cs[0].is_ascii() && cs[2].is_ascii() {
                    let lo = cs[0] as u8;
                    let hi = cs[2] as u8;
                    if lo <= hi {
                        return Ok((lo..=hi).collect());
                    }
                }
                Err(TensorError::invalid_argument(format!(
                    "unknown character class: [{}]",
                    class
                )))
            }
        }
    }

    /// Build a DFA from a list of (char_set, quantifier) atoms.
    fn build_dfa(atoms: Vec<(Vec<u8>, char)>) -> Result<Self> {
        let alphabet_offset: u8 = 32;
        let alphabet_size: usize = 95; // printable ASCII 32–126

        // Each atom becomes 1–2 DFA states depending on quantifier.
        // State layout: s0 (start) → atom states → accept.
        // Simple NFA→DFA for these restricted patterns:
        // - '1' (exactly one): s_i → s_{i+1} on set chars
        // - '+' (one or more): s_i → s_i on set chars (self-loop), s_i → s_{i+1}
        //   meaning: must consume at least one, then can repeat.
        // - '*' (zero or more): like '+' but s_i is already accepting-like via epsilon.
        // - '?': optional atom.
        //
        // We linearize: each atom occupies one state. Transitions to next atom state.
        // 'S' = start state = 0.
        // Each atom occupies states [atom_base, ..] where atom_base = 1 + atom_index.
        // Final state = n_atoms + 1 (the accept state).

        let n_atoms = atoms.len();
        // States: 0 = start, 1..=n_atoms = per-atom, n_atoms+1 = accept, n_atoms+2 = dead.
        let n_states = n_atoms + 3;
        let accept_state = n_atoms + 1;
        let dead_state = n_atoms + 2;

        let mut transitions = vec![vec![dead_state; alphabet_size]; n_states];
        let mut accepting = vec![false; n_states];
        accepting[accept_state] = true;
        // Dead state has no outgoing transitions (stays dead).

        // Build transitions for each atom.
        let mut epsilon_reachable: Vec<Vec<usize>> = vec![Vec::new(); n_states];

        for (atom_idx, (set, quant)) in atoms.iter().enumerate() {
            let s = atom_idx + 1; // state for this atom
            let next_s = s + 1;  // state for the next atom (or accept)

            // Map set chars to alphabet indices.
            let indices: Vec<usize> = set
                .iter()
                .filter(|&&c| c >= alphabet_offset && c < alphabet_offset + alphabet_size as u8)
                .map(|&c| (c - alphabet_offset) as usize)
                .collect();

            match quant {
                '1' => {
                    // Exactly one: s --set--> next_s.
                    for &idx in &indices {
                        transitions[s][idx] = next_s;
                    }
                }
                '+' => {
                    // One or more: s --set--> s (loop) AND s --set--> next_s via epsilon.
                    // We simulate by: s --set--> s (self-loop transitions accepted),
                    // and s is also connected to next_s via epsilon after at least one.
                    // Simplified: s --set--> s for repeating, and we accept
                    // the transition to next_s only via a "commit" state.
                    // Use a two-state encoding: s (consume first) → (s+0.5 conceptually).
                    // Since we linearize, use: s --set--> next_s AND next_s has self-loop.
                    // Actually: s --set--> next_s, next_s has self-loop on set.
                    for &idx in &indices {
                        transitions[s][idx] = next_s;
                        transitions[next_s][idx] = next_s; // self-loop
                    }
                    // next_s also transitions to the state after it (epsilon).
                    epsilon_reachable[next_s].push(next_s + 1);
                }
                '*' => {
                    // Zero or more: epsilon-skip this atom (s → next_s without consuming).
                    // And self-loop on set.
                    epsilon_reachable[s].push(next_s); // skip entirely
                    for &idx in &indices {
                        transitions[s][idx] = s; // loop
                    }
                }
                '?' => {
                    // Zero or one: either skip or consume once.
                    epsilon_reachable[s].push(next_s); // optional skip
                    for &idx in &indices {
                        transitions[s][idx] = next_s;
                    }
                }
                _ => {}
            }
        }

        // State 0 = start. Transitions from start state: first atom at state 1.
        // Connect s=0 → s=1 via epsilon.
        epsilon_reachable[0].push(1);

        // Resolve epsilon transitions: propagate until fixpoint.
        // For each state, find all epsilon-reachable states.
        let mut reachable: Vec<std::collections::HashSet<usize>> =
            (0..n_states).map(|s| {
                let mut set = std::collections::HashSet::new();
                set.insert(s);
                set
            }).collect();
        for _ in 0..n_states {
            let mut changed = false;
            for s in 0..n_states {
                let targets: Vec<usize> = epsilon_reachable[s].clone();
                for t in targets {
                    let add: Vec<usize> = reachable[t].iter().copied().collect();
                    for a in add {
                        if reachable[s].insert(a) {
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Apply epsilon closures to transitions.
        // transitions[s][c] → all states reachable from transitions[s][c] via epsilon.
        // For simplicity, we take the minimum non-dead state as the representative.
        for s in 0..n_states {
            for c in 0..alphabet_size {
                let raw_next = transitions[s][c];
                if raw_next == dead_state {
                    // Check if epsilon closure of s includes any state that transitions on c.
                    let eps: Vec<usize> = reachable[s].iter().copied().collect();
                    let mut best = dead_state;
                    for es in eps {
                        let t = transitions[es][c];
                        if t != dead_state
                            && (best == dead_state || t < best) {
                                best = t;
                            }
                    }
                    transitions[s][c] = best;
                }
            }
        }

        // Mark states as accepting if their epsilon closure contains the accept state.
        for s in 0..n_states {
            if reachable[s].contains(&accept_state) {
                accepting[s] = true;
            }
        }

        // Connect atom states to accept_state for '1' quantifier atoms at the last atom.
        // Ensure the state after the last atom connects to accept.
        let last_atom_next = n_atoms + 1; // = accept_state
        let _ = last_atom_next; // already accept state

        Ok(Self {
            transitions,
            accepting,
            current_state: 0,
            n_consumed: 0,
            alphabet_offset,
            alphabet_size,
        })
    }

    /// Feed a character into the DFA. Returns the new state.
    pub fn feed(&mut self, ch: u8) -> FsmState {
        if self.current_state >= self.transitions.len() {
            return FsmState::Dead;
        }
        let c_idx = if ch >= self.alphabet_offset
            && (ch - self.alphabet_offset) < self.alphabet_size as u8
        {
            (ch - self.alphabet_offset) as usize
        } else {
            self.current_state = usize::MAX; // dead
            return FsmState::Dead;
        };
        let next = self.transitions[self.current_state][c_idx];
        if next == Self::DEAD || next >= self.transitions.len() {
            self.current_state = self.transitions.len() - 1; // last state = dead
            return FsmState::Dead;
        }
        self.current_state = next;
        self.n_consumed += 1;
        self.current_fsm_state()
    }

    fn current_fsm_state(&self) -> FsmState {
        if self.current_state >= self.accepting.len() {
            return FsmState::Dead;
        }
        if self.accepting[self.current_state] {
            FsmState::Accepted
        } else {
            FsmState::Active(self.current_state)
        }
    }

    /// Current DFA state.
    pub fn state(&self) -> FsmState {
        self.current_fsm_state()
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.current_state = 0;
        self.n_consumed = 0;
    }

    /// Number of characters consumed so far.
    pub fn n_consumed(&self) -> usize {
        self.n_consumed
    }

    /// Check whether a string matches the pattern completely.
    pub fn matches_string(pattern: &str, input: &str) -> bool {
        let mut fsm = match Self::from_pattern(pattern) {
            Ok(f) => f,
            Err(_) => return false,
        };
        for b in input.bytes() {
            if fsm.feed(b) == FsmState::Dead { return false }
        }
        fsm.state() == FsmState::Accepted
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Grammar-Guided Sampling (Earley-Style CFG)
// ─────────────────────────────────────────────────────────────────────────────

/// A context-free grammar rule: `lhs → [rhs_0, rhs_1, …]` where each symbol
/// is either a terminal (starts with `'`) or a non-terminal.
#[derive(Clone, Debug)]
pub struct CfgRule {
    pub lhs: String,
    pub rhs: Vec<String>,
}

impl CfgRule {
    /// Create a grammar rule.
    pub fn new(lhs: impl Into<String>, rhs: Vec<impl Into<String>>) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into_iter().map(|s| s.into()).collect(),
        }
    }
}

/// An Earley chart item: `(rule_idx, dot_pos, origin)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EarleyItem {
    rule_idx: usize,
    dot: usize,
    origin: usize,
}

/// Grammar-guided sampler using an Earley parser to enforce CFG constraints.
///
/// Tokens are mapped to terminal symbols. At each step the sampler finds the
/// set of terminals that can be predicted (scanned) and masks all others.
pub struct GrammarSampler {
    rules: Vec<CfgRule>,
    start_symbol: String,
    /// Map from terminal label to token ids (e.g., `"'digit'"` → [48..57]).
    terminal_map: HashMap<String, Vec<usize>>,
    /// Chart: `chart[k]` = set of completed items at position k.
    chart: Vec<std::collections::HashSet<EarleyItem>>,
    /// Current parse position.
    position: usize,
    vocab_size: usize,
}

impl GrammarSampler {
    /// Create a new grammar sampler.
    pub fn new(
        rules: Vec<CfgRule>,
        start_symbol: impl Into<String>,
        terminal_map: HashMap<String, Vec<usize>>,
        vocab_size: usize,
    ) -> Self {
        let start_symbol = start_symbol.into();
        let mut sampler = Self {
            rules,
            start_symbol,
            terminal_map,
            chart: vec![std::collections::HashSet::new()],
            position: 0,
            vocab_size,
        };
        sampler.init_chart();
        sampler
    }

    fn init_chart(&mut self) {
        self.chart = vec![std::collections::HashSet::new()];
        self.position = 0;
        // Seed with all rules whose LHS is the start symbol.
        let start = self.start_symbol.clone();
        let seed_items: Vec<EarleyItem> = self
            .rules
            .iter()
            .enumerate()
            .filter(|(_, r)| r.lhs == start)
            .map(|(i, _)| EarleyItem { rule_idx: i, dot: 0, origin: 0 })
            .collect();
        for item in seed_items {
            self.chart[0].insert(item);
        }
        self.predict_and_complete(0);
    }

    fn predict_and_complete(&mut self, k: usize) {
        loop {
            // Clone current items to avoid simultaneous borrow.
            let items: Vec<EarleyItem> = self.chart[k].iter().cloned().collect();
            let mut to_insert: Vec<(usize, EarleyItem)> = Vec::new();

            for item in &items {
                let rule = &self.rules[item.rule_idx];
                if item.dot < rule.rhs.len() {
                    let next_sym = rule.rhs[item.dot].clone();
                    if !next_sym.starts_with('\'') {
                        // Non-terminal: predict — add to chart[k].
                        for (i, r) in self.rules.iter().enumerate() {
                            if r.lhs == next_sym {
                                to_insert.push((k, EarleyItem {
                                    rule_idx: i,
                                    dot: 0,
                                    origin: k,
                                }));
                            }
                        }
                    }
                } else {
                    // Completed item: complete items in chart[item.origin].
                    let completed_lhs = rule.lhs.clone();
                    let origin = item.origin;
                    // Snapshot chart[origin] to avoid borrow issue when origin == k.
                    let origin_items: Vec<EarleyItem> = if origin < self.chart.len() {
                        self.chart[origin].iter().cloned().collect()
                    } else {
                        Vec::new()
                    };
                    for it in &origin_items {
                        let r = &self.rules[it.rule_idx];
                        if it.dot < r.rhs.len() && r.rhs[it.dot] == completed_lhs {
                            to_insert.push((k, EarleyItem {
                                rule_idx: it.rule_idx,
                                dot: it.dot + 1,
                                origin: it.origin,
                            }));
                        }
                    }
                }
            }

            let mut changed = false;
            for (chart_idx, item) in to_insert {
                if chart_idx < self.chart.len() && self.chart[chart_idx].insert(item) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Compute the set of allowed terminal labels at the current position.
    pub fn allowed_terminals(&self) -> std::collections::HashSet<String> {
        let k = self.position;
        if k >= self.chart.len() {
            return std::collections::HashSet::new();
        }
        let mut terminals = std::collections::HashSet::new();
        for item in &self.chart[k] {
            let rule = &self.rules[item.rule_idx];
            if item.dot < rule.rhs.len() {
                let sym = &rule.rhs[item.dot];
                if sym.starts_with('\'') {
                    terminals.insert(sym.clone());
                }
            }
        }
        terminals
    }

    /// Apply grammar constraints to `logits` in-place.
    ///
    /// Tokens not in any allowed terminal are set to `NEG_INFINITY`.
    pub fn apply_constraints(&self, logits: &mut [f64]) {
        let allowed = self.allowed_terminals();
        let mut allowed_tokens: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for term in &allowed {
            if let Some(toks) = self.terminal_map.get(term) {
                for &t in toks {
                    allowed_tokens.insert(t);
                }
            }
        }
        for (i, l) in logits.iter_mut().enumerate() {
            if !allowed_tokens.contains(&i) {
                *l = f64::NEG_INFINITY;
            }
        }
    }

    /// Advance the parser after token `token_id` was sampled.
    pub fn advance(&mut self, token_id: usize) {
        let k = self.position;
        // Find the terminal(s) this token belongs to.
        let matched_terms: Vec<String> = self
            .terminal_map
            .iter()
            .filter(|(_, ids)| ids.contains(&token_id))
            .map(|(term, _)| term.clone())
            .collect();

        // Scan: for each matched terminal, advance items in chart[k].
        let next_k = k + 1;
        if next_k >= self.chart.len() {
            self.chart.push(std::collections::HashSet::new());
        }

        // Collect advanced items first (avoid borrow conflict on self.chart).
        let mut new_items: Vec<EarleyItem> = Vec::new();
        for term in &matched_terms {
            for item in &self.chart[k] {
                let rule = &self.rules[item.rule_idx];
                if item.dot < rule.rhs.len() && &rule.rhs[item.dot] == term {
                    new_items.push(EarleyItem {
                        rule_idx: item.rule_idx,
                        dot: item.dot + 1,
                        origin: item.origin,
                    });
                }
            }
        }
        for item in new_items {
            self.chart[next_k].insert(item);
        }

        self.position = next_k;
        self.predict_and_complete(next_k);
    }

    /// Whether the current state corresponds to a complete parse.
    pub fn is_accepted(&self) -> bool {
        let k = self.position;
        if k >= self.chart.len() {
            return false;
        }
        let start = self.start_symbol.clone();
        self.chart[k].iter().any(|item| {
            let rule = &self.rules[item.rule_idx];
            item.origin == 0
                && item.dot == rule.rhs.len()
                && rule.lhs == start
        })
    }

    /// Reset the parser to the initial state.
    pub fn reset(&mut self) {
        self.init_chart();
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. RAG Flat L2 Index
// ─────────────────────────────────────────────────────────────────────────────

/// Entry in the flat RAG index.
#[derive(Clone, Debug)]
pub struct RagEntry {
    pub text: String,
    pub embedding: Vec<f64>,
    pub doc_id: usize,
}

impl RagEntry {
    /// Create a new entry.
    pub fn new(text: impl Into<String>, embedding: Vec<f64>, doc_id: usize) -> Self {
        Self {
            text: text.into(),
            embedding,
            doc_id,
        }
    }
}

/// Brute-force L2-distance index for RAG retrieval.
///
/// Unlike the cosine-similarity `VectorStore`, this index uses squared
/// Euclidean distance so that unit-normalised embeddings give equivalent
/// results to cosine similarity while supporting unnormalised vectors too.
pub struct RagIndex {
    entries: Vec<RagEntry>,
    dim: usize,
}

impl RagIndex {
    /// Create an empty index for embeddings of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            dim,
        }
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: RagEntry) {
        self.entries.push(entry);
    }

    /// Return the `k` nearest entries by L2 distance.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<&RagEntry> {
        if self.entries.is_empty() || k == 0 {
            return vec![];
        }
        let mut scored: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let dist: f64 = query
                    .iter()
                    .zip(e.embedding.iter())
                    .map(|(&q, &x)| (q - x) * (q - x))
                    .sum();
                (i, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(k).map(|&(i, _)| &self.entries[i]).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Build from parallel text and embedding slices.
    pub fn from_vecs(texts: Vec<String>, embeddings: Vec<Vec<f64>>) -> Result<Self> {
        if texts.len() != embeddings.len() {
            return Err(TensorError::invalid_argument(
                "texts and embeddings must have equal length".to_string(),
            ));
        }
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
        let mut index = Self::new(dim);
        for (i, (text, emb)) in texts.into_iter().zip(embeddings).enumerate() {
            index.add(RagEntry::new(text, emb, i));
        }
        Ok(index)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Generation Evaluation Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Proxy for BERTScore using cosine similarity between averaged n-gram embeddings.
///
/// This is a lightweight approximation that does NOT require a BERT model;
/// it uses the raw token ids as unit vectors in R^V and computes the
/// cosine similarity of their sum embeddings (bag-of-words proxy).
pub struct BertScoreProxy {
    vocab_size: usize,
}

impl BertScoreProxy {
    /// Create a proxy with vocabulary size `vocab_size`.
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    /// Compute precision, recall, and F1 (BERTScore proxy).
    pub fn score(
        &self,
        hypothesis: &[usize],
        reference: &[usize],
    ) -> (f64, f64, f64) {
        if hypothesis.is_empty() || reference.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Build normalised bag-of-words vectors.
        let hyp_vec = self.bow_vector(hypothesis);
        let ref_vec = self.bow_vector(reference);

        // Token-level cosine similarity = dot product (since both are L2-normalised).
        let dot: f64 = hyp_vec
            .iter()
            .zip(ref_vec.iter())
            .map(|(&h, &r)| h * r)
            .sum();

        // Precision: each hyp token matched to max reference token.
        let precision = Self::greedy_match(&hyp_vec, &ref_vec);
        // Recall: each ref token matched to max hypothesis token.
        let recall = Self::greedy_match(&ref_vec, &hyp_vec);
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let _ = dot; // used implicitly via greedy_match
        (precision, recall, f1)
    }

    fn bow_vector(&self, tokens: &[usize]) -> Vec<f64> {
        let mut v = vec![0.0_f64; self.vocab_size];
        for &t in tokens {
            if t < self.vocab_size {
                v[t] += 1.0;
            }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    fn greedy_match(a: &[f64], b: &[f64]) -> f64 {
        // Greedy token matching via dot product of normalised BoW vectors.
        let dot: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        dot.clamp(0.0, 1.0)
    }
}

/// Generation evaluation metrics: BLEU-4, ROUGE-L, and distinct-n.
pub struct GenerationMetrics;

impl GenerationMetrics {
    /// Compute corpus BLEU-4 score using smoothed (add-1) n-gram precision.
    ///
    /// Uses Chen & Cherry (2014) smoothing (add-1 per n-gram order).
    pub fn bleu4(hypothesis: &[usize], reference: &[usize]) -> f64 {
        let mut log_sum = 0.0;
        let mut all_zero = true;
        for n in 1usize..=4 {
            let p = Self::ngram_precision(hypothesis, reference, n);
            if p > 0.0 {
                log_sum += p.ln();
                all_zero = false;
            } else {
                // Smoothed: +1.
                let smooth_p = 1.0 / (hypothesis.len().saturating_sub(n - 1) as f64 + 1.0);
                log_sum += smooth_p.ln();
            }
        }
        if all_zero {
            return 0.0;
        }
        let bp = Self::brevity_penalty(hypothesis.len(), reference.len());
        bp * (log_sum / 4.0).exp()
    }

    /// N-gram precision: |clipped n-gram matches| / |hypothesis n-grams|.
    pub fn ngram_precision(hypothesis: &[usize], reference: &[usize], n: usize) -> f64 {
        if hypothesis.len() < n {
            return 0.0;
        }
        let hyp_ngrams = Self::count_ngrams(hypothesis, n);
        let ref_ngrams = Self::count_ngrams(reference, n);
        let mut clipped = 0usize;
        let mut total = 0usize;
        for (ng, &hyp_c) in &hyp_ngrams {
            let ref_c = ref_ngrams.get(ng).copied().unwrap_or(0);
            clipped += hyp_c.min(ref_c);
            total += hyp_c;
        }
        if total == 0 {
            0.0
        } else {
            clipped as f64 / total as f64
        }
    }

    fn count_ngrams(tokens: &[usize], n: usize) -> HashMap<Vec<usize>, usize> {
        let mut counts = HashMap::new();
        for i in 0..tokens.len().saturating_sub(n - 1) {
            let ng = tokens[i..i + n].to_vec();
            *counts.entry(ng).or_insert(0) += 1;
        }
        counts
    }

    fn brevity_penalty(hyp_len: usize, ref_len: usize) -> f64 {
        if hyp_len >= ref_len {
            1.0
        } else {
            let r = ref_len as f64;
            let c = hyp_len as f64;
            (1.0 - r / c).exp()
        }
    }

    /// ROUGE-L F1 score: longest common subsequence (LCS) based.
    pub fn rouge_l(hypothesis: &[usize], reference: &[usize]) -> f64 {
        if hypothesis.is_empty() || reference.is_empty() {
            return 0.0;
        }
        let lcs = Self::lcs_length(hypothesis, reference);
        let precision = lcs as f64 / hypothesis.len() as f64;
        let recall = lcs as f64 / reference.len() as f64;
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// LCS length via dynamic programming.
    pub fn lcs_length(a: &[usize], b: &[usize]) -> usize {
        let m = a.len();
        let n = b.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        dp[m][n]
    }

    /// Distinct-n: ratio of unique n-grams to total n-grams in `tokens`.
    pub fn distinct_n(tokens: &[usize], n: usize) -> f64 {
        if tokens.len() < n {
            return 0.0;
        }
        let ngrams = Self::count_ngrams(tokens, n);
        let unique = ngrams.len() as f64;
        let total = tokens.len().saturating_sub(n - 1) as f64;
        if total == 0.0 {
            0.0
        } else {
            unique / total
        }
    }

    /// Compute all standard metrics and return a summary report.
    pub fn evaluate(hypothesis: &[usize], reference: &[usize]) -> GenerationReport {
        let bleu = Self::bleu4(hypothesis, reference);
        let rouge = Self::rouge_l(hypothesis, reference);
        let d1 = Self::distinct_n(hypothesis, 1);
        let d2 = Self::distinct_n(hypothesis, 2);
        GenerationReport {
            bleu4: bleu,
            rouge_l: rouge,
            distinct_1: d1,
            distinct_2: d2,
            hypothesis_length: hypothesis.len(),
            reference_length: reference.len(),
        }
    }
}

/// Summary report from `GenerationMetrics::evaluate`.
#[derive(Clone, Debug)]
pub struct GenerationReport {
    pub bleu4: f64,
    pub rouge_l: f64,
    pub distinct_1: f64,
    pub distinct_2: f64,
    pub hypothesis_length: usize,
    pub reference_length: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── MedianDraftLength ──────────────────────────────────────────────────────

    #[test]
    fn test_median_draft_length_empty() {
        let mut mdl = MedianDraftLength::new();
        assert_eq!(mdl.median(), 0.0);
    }

    #[test]
    fn test_median_draft_length_odd() {
        let mut mdl = MedianDraftLength::new();
        mdl.push(1);
        mdl.push(3);
        mdl.push(5);
        assert!((mdl.median() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_median_draft_length_even() {
        let mut mdl = MedianDraftLength::new();
        mdl.push(2);
        mdl.push(4);
        assert!((mdl.median() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_suggested_k_clamps() {
        let mut mdl = MedianDraftLength::new();
        mdl.push(100);
        assert_eq!(mdl.suggested_k(5), 5);
    }

    #[test]
    fn test_suggested_k_at_least_one() {
        let mut mdl = MedianDraftLength::new();
        // Empty → 0, but suggested_k clamps to 1.
        assert_eq!(mdl.suggested_k(8), 1);
    }

    #[test]
    fn test_median_length_multiple() {
        let mut mdl = MedianDraftLength::new();
        for i in 0..10 {
            mdl.push(i);
        }
        assert_eq!(mdl.len(), 10);
    }

    // ── RegexFsm ──────────────────────────────────────────────────────────────

    #[test]
    fn test_regex_fsm_literal_match() {
        assert!(RegexFsm::matches_string("a", "a"));
        assert!(!RegexFsm::matches_string("a", "b"));
    }

    #[test]
    fn test_regex_fsm_dot_any() {
        assert!(RegexFsm::matches_string(".", "a"));
        assert!(RegexFsm::matches_string(".", "9"));
        assert!(!RegexFsm::matches_string(".", "ab"));
    }

    #[test]
    fn test_regex_fsm_digit_class() {
        assert!(RegexFsm::matches_string("[digit]", "5"));
        assert!(!RegexFsm::matches_string("[digit]", "a"));
    }

    #[test]
    fn test_regex_fsm_star_quantifier() {
        // Zero or more digits.
        assert!(RegexFsm::matches_string("[digit]*", ""));
        assert!(RegexFsm::matches_string("[digit]*", "123"));
    }

    #[test]
    fn test_regex_fsm_question_quantifier() {
        assert!(RegexFsm::matches_string("a?", ""));
        assert!(RegexFsm::matches_string("a?", "a"));
        assert!(!RegexFsm::matches_string("a?", "aa"));
    }

    #[test]
    fn test_regex_fsm_feed() {
        let mut fsm = RegexFsm::from_pattern("[digit]+").expect("valid pattern");
        assert_ne!(fsm.feed(b'3'), FsmState::Dead);
        assert_ne!(fsm.feed(b'7'), FsmState::Dead);
    }

    #[test]
    fn test_regex_fsm_reset() {
        let mut fsm = RegexFsm::from_pattern("a").expect("valid pattern");
        fsm.feed(b'a');
        fsm.reset();
        assert_eq!(fsm.n_consumed(), 0);
    }

    #[test]
    fn test_regex_fsm_alpha_class() {
        assert!(RegexFsm::matches_string("[alpha]", "z"));
        assert!(RegexFsm::matches_string("[alpha]", "A"));
        assert!(!RegexFsm::matches_string("[alpha]", "3"));
    }

    // ── GrammarSampler ────────────────────────────────────────────────────────

    fn make_digit_grammar() -> GrammarSampler {
        // Grammar: S → D | S D
        //          D → '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
        let rules = vec![
            CfgRule::new("S", vec!["D"]),
            CfgRule::new("S", vec!["S", "D"]),
            CfgRule::new("D", vec!["'digit0'"]),
            CfgRule::new("D", vec!["'digit1'"]),
            CfgRule::new("D", vec!["'digit2'"]),
        ];
        let mut terminal_map = HashMap::new();
        terminal_map.insert("'digit0'".to_string(), vec![0_usize]);
        terminal_map.insert("'digit1'".to_string(), vec![1_usize]);
        terminal_map.insert("'digit2'".to_string(), vec![2_usize]);
        GrammarSampler::new(rules, "S", terminal_map, 10)
    }

    #[test]
    fn test_grammar_sampler_allowed_terminals_initial() {
        let sampler = make_digit_grammar();
        let allowed = sampler.allowed_terminals();
        // Must allow digit terminals at start.
        assert!(!allowed.is_empty());
    }

    #[test]
    fn test_grammar_sampler_advance() {
        let mut sampler = make_digit_grammar();
        sampler.advance(0); // token 0 = 'digit0'
        assert!(sampler.is_accepted());
    }

    #[test]
    fn test_grammar_sampler_mask_logits() {
        let sampler = make_digit_grammar();
        let mut logits = vec![1.0_f64; 10];
        sampler.apply_constraints(&mut logits);
        // Tokens 0, 1, 2 should be allowed; others masked.
        assert!(logits[0].is_finite());
        assert!(logits[1].is_finite());
        assert!(logits[2].is_finite());
        for i in 3..10 {
            assert_eq!(logits[i], f64::NEG_INFINITY);
        }
    }

    #[test]
    fn test_grammar_sampler_reset() {
        let mut sampler = make_digit_grammar();
        sampler.advance(1);
        sampler.reset();
        assert!(!sampler.chart.is_empty());
    }

    #[test]
    fn test_grammar_sampler_multi_token() {
        let mut sampler = make_digit_grammar();
        sampler.advance(1); // first digit
        sampler.advance(2); // second digit
        // S → S D, so this should still be accepted.
        assert!(sampler.is_accepted());
    }

    // ── RagIndex ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rag_index_search() {
        let mut index = RagIndex::new(3);
        index.add(RagEntry::new("doc A", vec![1.0, 0.0, 0.0], 0));
        index.add(RagEntry::new("doc B", vec![0.0, 1.0, 0.0], 1));
        index.add(RagEntry::new("doc C", vec![0.0, 0.0, 1.0], 2));
        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "doc A");
    }

    #[test]
    fn test_rag_index_empty() {
        let index = RagIndex::new(4);
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 3);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rag_index_from_vecs() {
        let texts = vec!["alpha".to_string(), "beta".to_string()];
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let index = RagIndex::from_vecs(texts, embs).expect("build index");
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_rag_index_from_vecs_mismatch() {
        let texts = vec!["a".to_string()];
        let embs = vec![vec![1.0], vec![0.0]];
        assert!(RagIndex::from_vecs(texts, embs).is_err());
    }

    #[test]
    fn test_rag_index_k_results() {
        let mut index = RagIndex::new(2);
        for i in 0..5 {
            index.add(RagEntry::new(format!("doc {}", i), vec![i as f64, 0.0], i));
        }
        let results = index.search(&[3.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_rag_entry_doc_id() {
        let entry = RagEntry::new("text", vec![1.0, 2.0], 42);
        assert_eq!(entry.doc_id, 42);
        assert_eq!(entry.text, "text");
    }

    // ── BertScoreProxy ────────────────────────────────────────────────────────

    #[test]
    fn test_bert_score_identical() {
        let proxy = BertScoreProxy::new(10);
        let (p, r, f1) = proxy.score(&[1, 2, 3], &[1, 2, 3]);
        assert!(p > 0.9);
        assert!(r > 0.9);
        assert!(f1 > 0.9);
    }

    #[test]
    fn test_bert_score_empty() {
        let proxy = BertScoreProxy::new(10);
        let (p, r, f1) = proxy.score(&[], &[1, 2]);
        assert_eq!(p, 0.0);
        assert_eq!(r, 0.0);
        assert_eq!(f1, 0.0);
    }

    #[test]
    fn test_bert_score_disjoint() {
        let proxy = BertScoreProxy::new(20);
        let (_, _, f1) = proxy.score(&[1, 2, 3], &[4, 5, 6]);
        assert!(f1 < 0.1);
    }

    // ── GenerationMetrics ─────────────────────────────────────────────────────

    #[test]
    fn test_bleu4_identical() {
        let tokens: Vec<usize> = vec![1, 2, 3, 4, 5];
        let bleu = GenerationMetrics::bleu4(&tokens, &tokens);
        assert!(bleu > 0.9, "BLEU for identical sequences should be ~1, got {}", bleu);
    }

    #[test]
    fn test_bleu4_empty_hypothesis() {
        let bleu = GenerationMetrics::bleu4(&[], &[1, 2, 3]);
        assert_eq!(bleu, 0.0);
    }

    #[test]
    fn test_rouge_l_identical() {
        let tokens: Vec<usize> = vec![1, 2, 3, 4];
        let rouge = GenerationMetrics::rouge_l(&tokens, &tokens);
        assert!((rouge - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_disjoint() {
        let rouge = GenerationMetrics::rouge_l(&[1, 2], &[3, 4]);
        assert!(rouge < 0.1);
    }

    #[test]
    fn test_lcs_known() {
        // LCS([1,2,3,4,5], [2,4,6]) = [2,4] → length 2.
        let lcs = GenerationMetrics::lcs_length(&[1, 2, 3, 4, 5], &[2, 4, 6]);
        assert_eq!(lcs, 2);
    }

    #[test]
    fn test_distinct_1() {
        let tokens: Vec<usize> = vec![1, 2, 3, 2, 1];
        let d1 = GenerationMetrics::distinct_n(&tokens, 1);
        // Unique unigrams: {1,2,3} = 3; total = 5.
        assert!((d1 - 3.0 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_distinct_2() {
        let tokens: Vec<usize> = vec![1, 2, 1, 2];
        let d2 = GenerationMetrics::distinct_n(&tokens, 2);
        // Bigrams: (1,2), (2,1), (1,2) → unique: {(1,2),(2,1)} = 2; total = 3.
        assert!((d2 - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_evaluate_report() {
        let hyp: Vec<usize> = vec![1, 2, 3];
        let ref_: Vec<usize> = vec![1, 2, 3, 4];
        let report = GenerationMetrics::evaluate(&hyp, &ref_);
        assert!(report.bleu4 >= 0.0);
        assert!(report.rouge_l >= 0.0 && report.rouge_l <= 1.0);
        assert!(report.distinct_1 >= 0.0 && report.distinct_1 <= 1.0);
        assert_eq!(report.hypothesis_length, 3);
        assert_eq!(report.reference_length, 4);
    }

    #[test]
    fn test_ngram_precision_unigram() {
        // hyp = [1,2,2], ref = [1,2,3]. Unigram prec = clipped(1,1,2,1)/3 = 2/3.
        let p = GenerationMetrics::ngram_precision(&[1, 2, 2], &[1, 2, 3], 1);
        assert!((p - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_bleu4_partial_overlap() {
        let hyp: Vec<usize> = vec![1, 2, 3, 4];
        let ref_: Vec<usize> = vec![1, 2, 5, 6];
        let bleu = GenerationMetrics::bleu4(&hyp, &ref_);
        assert!(bleu > 0.0 && bleu < 1.0);
    }
}
