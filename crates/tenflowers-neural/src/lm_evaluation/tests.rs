use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// §1  LmePerplexity tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_perplexity_basic() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    // log P = -1 for each token → NLL = 1 → PPL = e^1 ≈ 2.718
    let token_ids = vec![1u32, 2, 3, 4];
    let log_probs = vec![-1.0f64; 4];
    let result = ppl.compute(&token_ids, &log_probs).expect("perplexity");
    assert!((result - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_lme_perplexity_uniform() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    // log P = -2.0 → PPL = e^2
    let token_ids = vec![0u32; 10];
    let log_probs = vec![-2.0f64; 10];
    let result = ppl.compute(&token_ids, &log_probs).expect("ppl");
    assert!((result - (2.0f64).exp()).abs() < 1e-8);
}

#[test]
fn test_lme_perplexity_empty_error() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    assert!(ppl.compute(&[], &[]).is_err());
}

#[test]
fn test_lme_perplexity_length_mismatch() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    assert!(ppl.compute(&[1, 2], &[-1.0]).is_err());
}

#[test]
fn test_lme_perplexity_sliding_window() {
    let cfg = LmePerplexityConfig {
        stride: 3,
        byte_pair: false,
    };
    let ppl = LmePerplexity::new(cfg);
    let token_ids: Vec<u32> = (0..12).collect();
    let log_probs = vec![-0.5f64; 12];
    let result = ppl.compute(&token_ids, &log_probs).expect("sliding ppl");
    // PPL should be exp(0.5) ≈ 1.649
    assert!((result - (0.5f64).exp()).abs() < 0.01);
}

#[test]
fn test_lme_perplexity_byte_pair() {
    let cfg = LmePerplexityConfig {
        stride: 0,
        byte_pair: true,
    };
    let ppl = LmePerplexity::new(cfg);
    let token_ids = vec![0u32; 5];
    let log_probs = vec![-1.0f64; 5];
    let result = ppl.compute(&token_ids, &log_probs).expect("bpp");
    // BPP = exp(NLL / ln(4)) = exp(1 / ln(4))
    let expected = (1.0 / 4.0f64.ln()).exp();
    assert!((result - expected).abs() < 1e-8);
}

#[test]
fn test_lme_perplexity_word() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    let log_probs = vec![-10.0, -5.0];
    let word_counts = vec![3usize, 5];
    let result = ppl
        .word_perplexity(&log_probs, &word_counts)
        .expect("word ppl");
    // NLL = (10 + 5) / 8 = 1.875 → PPL = exp(1.875)
    let expected = (1.875f64).exp();
    assert!((result - expected).abs() < 1e-8);
}

#[test]
fn test_lme_perplexity_word_empty_error() {
    let ppl = LmePerplexity::new(LmePerplexityConfig::default());
    assert!(ppl.word_perplexity(&[], &[]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  LmeBLEU tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_bleu_perfect_match() {
    let bleu = LmeBLEU::new(4).expect("bleu");
    let result = bleu
        .compute_sentence("the cat is on the mat", &["the cat is on the mat"])
        .expect("bleu result");
    // Perfect match: BP = 1, all precisions = 1 → BLEU = 1
    assert!(result.bleu > 0.99);
    assert!((result.brevity_penalty - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_bleu_empty_hypothesis() {
    let bleu = LmeBLEU::new(2).expect("bleu");
    let result = bleu.compute_sentence("a b c", &["x y z"]).expect("bleu");
    // No overlap
    assert!(result.bleu < 0.1);
}

#[test]
fn test_lme_bleu_brevity_penalty() {
    let bleu = LmeBLEU::new(1).expect("bleu");
    let result = bleu
        .compute_sentence("the", &["the cat is on the mat"])
        .expect("bleu");
    // BP < 1 because hypothesis is shorter
    assert!(result.brevity_penalty < 1.0);
}

#[test]
fn test_lme_bleu_invalid_max_n() {
    assert!(LmeBLEU::new(0).is_err());
}

#[test]
fn test_lme_bleu_multiple_references() {
    let bleu = LmeBLEU::new(2).expect("bleu");
    let hyp = "the cat sat on the mat";
    let refs = ["the cat is on the mat", "there is a cat on the mat"];
    let result = bleu.compute_sentence(hyp, &refs).expect("bleu");
    assert!(result.bleu > 0.0);
}

#[test]
fn test_lme_bleu_corpus() {
    let bleu = LmeBLEU::new(2).expect("bleu");
    let hyps = ["hello world", "good morning"];
    let refs: Vec<Vec<&str>> = vec![vec!["hello world"], vec!["good morning everyone"]];
    let refs_slice: Vec<&[&str]> = refs.iter().map(|r| r.as_slice()).collect();
    let result = bleu
        .compute_corpus(&hyps, &refs_slice)
        .expect("corpus bleu");
    assert!(result.bleu > 0.0);
}

#[test]
fn test_lme_bleu_mismatched_lengths() {
    let bleu = LmeBLEU::new(2).expect("bleu");
    let result = bleu.compute_corpus(&["a"], &[&["b"][..], &["c"][..]]);
    assert!(result.is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  LmeROUGE tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_rouge_n_perfect() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_n("the cat sat on the mat", "the cat sat on the mat", 1)
        .expect("rouge-n");
    assert!((score.f1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_rouge_n_no_overlap() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_n("apple orange", "cat dog", 1)
        .expect("rouge-n");
    assert!(score.f1 < 1e-10);
}

#[test]
fn test_lme_rouge_n_invalid_n() {
    let rouge = LmeROUGE::new(1.0);
    assert!(rouge.compute_rouge_n("a b", "a b", 0).is_err());
}

#[test]
fn test_lme_rouge_n_bigram() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_n("the cat sat on the mat", "the cat is on the mat", 2)
        .expect("rouge-2");
    assert!(score.f1 > 0.0 && score.f1 <= 1.0);
}

#[test]
fn test_lme_rouge_l_perfect() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_l("the cat sat on the mat", "the cat sat on the mat")
        .expect("rouge-l");
    assert!((score.f1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_rouge_l_empty_both() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge.compute_rouge_l("", "").expect("rouge-l empty");
    assert_eq!(score.f1, 1.0);
}

#[test]
fn test_lme_rouge_l_one_empty() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge.compute_rouge_l("hello", "").expect("rouge-l");
    assert_eq!(score.f1, 0.0);
}

#[test]
fn test_lme_rouge_w_basic() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_w("the quick brown fox", "the quick brown fox jumps")
        .expect("rouge-w");
    assert!(score.f1 > 0.0 && score.f1 <= 1.0);
}

#[test]
fn test_lme_rouge_partial_overlap() {
    let rouge = LmeROUGE::new(1.0);
    let score = rouge
        .compute_rouge_n("the cat", "the dog", 1)
        .expect("rouge-n");
    // "the" overlaps
    assert!(score.f1 > 0.0 && score.f1 < 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  LmeBERTScore tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_bertscore_identical() {
    let bs = LmeBERTScore::new(64).expect("bertscore");
    let result = bs.compute("hello world", "hello world").expect("bertscore");
    // Identical strings → all cosine sims = 1.0
    assert!((result.f1 - 1.0).abs() < 1e-8);
}

#[test]
fn test_lme_bertscore_empty_both() {
    let bs = LmeBERTScore::new(32).expect("bertscore");
    let result = bs.compute("", "").expect("bertscore");
    assert_eq!(result.f1, 1.0);
}

#[test]
fn test_lme_bertscore_one_empty() {
    let bs = LmeBERTScore::new(32).expect("bertscore");
    let result = bs.compute("hello", "").expect("bertscore");
    assert_eq!(result.f1, 0.0);
}

#[test]
fn test_lme_bertscore_different_tokens() {
    let bs = LmeBERTScore::new(16).expect("bertscore");
    let result = bs
        .compute("apple orange mango", "banana grape lemon")
        .expect("bertscore");
    // Different tokens → some sim but not 1.0
    assert!(result.f1 >= 0.0 && result.f1 <= 1.0);
}

#[test]
fn test_lme_bertscore_invalid_dim() {
    assert!(LmeBERTScore::new(0).is_err());
}

#[test]
fn test_lme_bertscore_f1_between_p_r() {
    let bs = LmeBERTScore::new(8).expect("bertscore");
    let result = bs
        .compute("the cat sat on the mat", "the cat is on a mat")
        .expect("bertscore");
    // F1 should be harmonic mean of P and R
    if result.precision + result.recall > 1e-12 {
        let expected_f1 =
            2.0 * result.precision * result.recall / (result.precision + result.recall);
        assert!((result.f1 - expected_f1).abs() < 1e-6);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  LmeMathEval tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_math_exact_match() {
    let eval = LmeMathEval::new(1e-6, false);
    let result = eval
        .evaluate_math_answer("the answer is 42", 42.0)
        .expect("math eval");
    assert!(result.is_correct);
    assert_eq!(result.extracted_answer, Some(42.0));
}

#[test]
fn test_lme_math_approximate_match() {
    let eval = LmeMathEval::new(0.01, true);
    let result = eval
        .evaluate_math_answer("approximately 3.14", std::f64::consts::PI)
        .expect("math eval");
    // 3.14 and 3.14159 differ by ~0.0016 < 0.01 → correct
    assert!(result.is_correct);
}

#[test]
fn test_lme_math_no_number() {
    let eval = LmeMathEval::new(1e-6, false);
    let result = eval
        .evaluate_math_answer("no numbers here!", 5.0)
        .expect("math eval");
    assert!(!result.is_correct);
    assert_eq!(result.extracted_answer, None);
}

#[test]
fn test_lme_math_cot_keywords() {
    let eval = LmeMathEval::new(1e-6, false);
    assert!(eval.check_cot_keywords("step 1: therefore 2 + 2 = 4"));
    assert!(!eval.check_cot_keywords("the number is 42"));
}

#[test]
fn test_lme_math_negative_number() {
    let eval = LmeMathEval::new(1e-6, false);
    let result = eval
        .evaluate_math_answer("the result is -7", -7.0)
        .expect("math eval");
    assert!(result.is_correct);
    assert_eq!(result.extracted_answer, Some(-7.0));
}

#[test]
fn test_lme_math_batch() {
    let eval = LmeMathEval::new(0.01, true);
    let preds = ["answer: 10", "about 3.14", "nothing"];
    let golds = [10.0, std::f64::consts::PI, 0.0];
    let (correct, total) = eval.batch_evaluate(&preds, &golds).expect("batch");
    assert_eq!(total, 3);
    assert_eq!(correct, 2); // "nothing" fails
}

#[test]
fn test_lme_math_batch_mismatch() {
    let eval = LmeMathEval::new(1e-6, false);
    assert!(eval.batch_evaluate(&["a"], &[1.0, 2.0]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  LmeFewShotEval tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_fewshot_build_prompt() {
    let cfg = LmeFewShotConfig {
        n_shots: 2,
        separator: "\n\n".to_string(),
        answer_prefix: "A:".to_string(),
    };
    let eval = LmeFewShotEval::new(cfg);
    let examples = vec![
        LmeFewShotExample {
            input: "Q: What is 2+2?".to_string(),
            output: "4".to_string(),
        },
        LmeFewShotExample {
            input: "Q: What is 3+3?".to_string(),
            output: "6".to_string(),
        },
    ];
    let prompt = eval.build_prompt(&examples, "Q: What is 4+4?");
    assert!(prompt.contains("Q: What is 4+4?"));
    assert!(prompt.contains("A: 4"));
    assert!(prompt.contains("A: 6"));
}

#[test]
fn test_lme_fewshot_evaluate_mc() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    let choices = ["Paris", "London", "Berlin"];
    let log_probs = [-0.1, -0.5, -0.8];
    let predicted = eval.evaluate_mc("", &choices, &log_probs).expect("mc");
    assert_eq!(predicted, 0); // Highest log-prob is choices[0]
}

#[test]
fn test_lme_fewshot_mc_empty_choices() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    assert!(eval.evaluate_mc("", &[], &[]).is_err());
}

#[test]
fn test_lme_fewshot_mc_mismatch() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    assert!(eval.evaluate_mc("", &["a", "b"], &[-1.0]).is_err());
}

#[test]
fn test_lme_fewshot_normalised_log_probs() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    let lp = [-4.0, -6.0];
    let tc = [2usize, 3];
    let norm = eval.normalised_log_probs(&lp, &tc).expect("norm");
    assert!((norm[0] - (-2.0)).abs() < 1e-10);
    assert!((norm[1] - (-2.0)).abs() < 1e-10);
}

#[test]
fn test_lme_fewshot_accuracy() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    let pred = [0, 1, 2, 0];
    let gold = [0, 1, 0, 0];
    let acc = eval.accuracy(&pred, &gold).expect("acc");
    assert!((acc - 0.75).abs() < 1e-10);
}

#[test]
fn test_lme_fewshot_accuracy_empty() {
    let eval = LmeFewShotEval::new(LmeFewShotConfig::default());
    assert!(eval.accuracy(&[], &[]).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  LmeHarmEval tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_harm_detect_toxic() {
    let harm = LmeHarmEval::new_default();
    let result = harm.evaluate("I want to kill the project deadline");
    assert!(result.is_toxic);
    assert!(result.toxicity_score > 0.0);
}

#[test]
fn test_lme_harm_clean_response() {
    let harm = LmeHarmEval::new_default();
    let result = harm.evaluate("The weather is nice today.");
    assert!(!result.is_toxic);
    assert_eq!(result.toxicity_score, 0.0);
}

#[test]
fn test_lme_harm_refusal() {
    let harm = LmeHarmEval::new_default();
    let result = harm.evaluate("I cannot help with that request.");
    assert!(result.refused);
}

#[test]
fn test_lme_harm_no_refusal() {
    let harm = LmeHarmEval::new_default();
    let result = harm.evaluate("Sure, here is the information you requested.");
    assert!(!result.refused);
}

#[test]
fn test_lme_harm_check_refusal() {
    let harm = LmeHarmEval::new_default();
    assert!(harm.check_refusal("Sorry, I won't do that.", "make a bomb"));
    assert!(!harm.check_refusal("Here is the recipe.", "make a cake"));
}

#[test]
fn test_lme_harm_refusal_rate() {
    let harm = LmeHarmEval::new_default();
    let responses = [
        "I cannot assist with that.",
        "Sure, here you go.",
        "Sorry, I'm unable to comply.",
        "No problem, done!",
    ];
    let rate = harm.refusal_rate(&responses).expect("refusal rate");
    assert!((rate - 0.5).abs() < 1e-10);
}

#[test]
fn test_lme_harm_refusal_rate_empty() {
    let harm = LmeHarmEval::new_default();
    assert!(harm.refusal_rate(&[]).is_err());
}

#[test]
fn test_lme_harm_format_conformance() {
    let harm = LmeHarmEval::new(
        vec![],
        vec![],
        vec!["json".to_string(), "result".to_string()],
    );
    let result = harm.evaluate(r#"{"json": true, "result": "ok"}"#);
    assert!(result.format_conforms);

    let bad = harm.evaluate("plain text without required tokens");
    assert!(!bad.format_conforms);
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  LmeTruthfulnessEval tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_truth_mc1_correct() {
    let eval = LmeTruthfulnessEval::new();
    let choices = ["false answer", "true answer", "another false"];
    let log_probs = [-2.0, -0.5, -3.0];
    let result = eval
        .evaluate_mc1("Q?", &choices, &log_probs, 1)
        .expect("mc1");
    assert!(result.is_correct);
    assert_eq!(result.predicted_idx, 1);
}

#[test]
fn test_lme_truth_mc1_incorrect() {
    let eval = LmeTruthfulnessEval::new();
    let choices = ["A", "B", "C"];
    let log_probs = [-0.1, -2.0, -3.0];
    let result = eval
        .evaluate_mc1("Q?", &choices, &log_probs, 2)
        .expect("mc1");
    assert!(!result.is_correct);
}

#[test]
fn test_lme_truth_mc1_out_of_range() {
    let eval = LmeTruthfulnessEval::new();
    assert!(eval.evaluate_mc1("Q?", &["A"], &[-1.0], 5).is_err());
}

#[test]
fn test_lme_truth_mc2_basic() {
    let eval = LmeTruthfulnessEval::new();
    let log_probs = [-0.5, -1.0, -2.0, -0.8];
    let truth_mask = [true, false, true, false];
    let result = eval
        .evaluate_mc2("Q?", &log_probs, &truth_mask)
        .expect("mc2");
    // True answers: indices 0 and 2; False: 1 and 3
    assert!(result.p_true > 0.0);
    assert!(result.p_false > 0.0);
    assert!((result.p_true + result.p_false - 1.0).abs() < 1e-6);
}

#[test]
fn test_lme_truth_mc2_all_true() {
    let eval = LmeTruthfulnessEval::new();
    let log_probs = [-1.0, -1.0];
    let truth_mask = [true, true];
    let result = eval
        .evaluate_mc2("Q?", &log_probs, &truth_mask)
        .expect("mc2");
    assert!((result.p_true - 1.0).abs() < 1e-6);
}

#[test]
fn test_lme_truth_mc2_mismatch() {
    let eval = LmeTruthfulnessEval::new();
    assert!(eval.evaluate_mc2("Q?", &[-1.0], &[true, false]).is_err());
}

#[test]
fn test_lme_truth_judge() {
    let eval = LmeTruthfulnessEval::new();
    assert!(eval.judge_truthfulness(
        "Is the sky blue?",
        "Yes, the sky is blue because of Rayleigh scattering.",
        &["yes", "blue"],
        &["no", "wrong"],
    ));
    assert!(!eval.judge_truthfulness(
        "Is the sky green?",
        "No it is wrong and false",
        &["yes", "correct"],
        &["no", "wrong", "false"],
    ));
}

#[test]
fn test_lme_truth_batch_mc1() {
    let eval = LmeTruthfulnessEval::new();
    let questions = ["Q1?", "Q2?"];
    let choices_list: Vec<Vec<&str>> = vec![vec!["A", "B"], vec!["X", "Y", "Z"]];
    let lp_list: Vec<Vec<f64>> = vec![vec![-0.1, -2.0], vec![-3.0, -0.2, -1.5]];
    let truth_indices = [0usize, 1];
    let acc = eval
        .batch_mc1_accuracy(&questions, &choices_list, &lp_list, &truth_indices)
        .expect("batch mc1");
    // Q1: predicted 0 = truth 0 ✓; Q2: predicted 1 = truth 1 ✓ → 100%
    assert!((acc - 1.0).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  LmeCodeEval tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_code_pass_at_k_zero_correct() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    let p = eval.pass_at_k(10, 0, 1).expect("pass@k");
    assert!((p - 0.0).abs() < 1e-10);
}

#[test]
fn test_lme_code_pass_at_k_all_correct() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    let p = eval.pass_at_k(10, 10, 1).expect("pass@k");
    assert!((p - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_code_pass_at_k_unbiased() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    // n=100, c=50, k=10 → should be reasonably high
    let p = eval.pass_at_k(100, 50, 10).expect("pass@k");
    assert!(p > 0.99); // Very likely to find one correct in 10
}

#[test]
fn test_lme_code_pass_at_k_invalid() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    assert!(eval.pass_at_k(0, 0, 1).is_err());
    assert!(eval.pass_at_k(5, 10, 1).is_err()); // c > n
}

#[test]
fn test_lme_code_pass_at_k_invalid_k() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    assert!(eval.pass_at_k(10, 5, 0).is_err());
}

#[test]
fn test_lme_code_batch_pass() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    let n_list = [10usize, 10, 10];
    let c_list = [10usize, 5, 0];
    let avg = eval.batch_pass_at_k(&n_list, &c_list).expect("batch pass");
    // (1.0 + p50 + 0.0) / 3
    assert!(avg > 0.0 && avg <= 1.0);
}

#[test]
fn test_lme_code_syntax_valid_balanced() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    assert!(eval.check_syntax_validity("fn foo() { let x = vec![1, 2, 3]; }"));
}

#[test]
fn test_lme_code_syntax_invalid_unbalanced() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    assert!(!eval.check_syntax_validity("fn foo() { let x = vec![1, 2, 3];"));
}

#[test]
fn test_lme_code_syntax_empty() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    assert!(eval.check_syntax_validity(""));
}

#[test]
fn test_lme_code_simulate_execution_match() {
    let eval = LmeCodeEval::new(1).expect("code eval");
    // Code contains "42" which matches gold 42.0
    let ok = eval.simulate_execution("result = 42", 42.0).expect("sim");
    assert!(ok);
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  LmeReport tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_report_new() {
    let report = LmeReport::new("gpt-test");
    assert_eq!(report.model_a, "gpt-test");
    assert!(report.perplexity_a.is_none());
}

#[test]
fn test_lme_report_bootstrap_ci() {
    let mut report = LmeReport::new("model-a");
    let values = vec![0.8, 0.82, 0.78, 0.81, 0.79, 0.83, 0.80, 0.79, 0.81, 0.82];
    let (lo, hi) = report
        .bootstrap_ci("bleu", &values, 1000, 0.05)
        .expect("ci");
    assert!(lo < hi);
    assert!(lo > 0.0 && hi <= 1.0);
    assert!(report.bootstrap_ci.contains_key("bleu"));
}

#[test]
fn test_lme_report_bootstrap_ci_invalid_alpha() {
    let mut report = LmeReport::new("m");
    assert!(report.bootstrap_ci("x", &[1.0], 100, 0.0).is_err());
    assert!(report.bootstrap_ci("x", &[1.0], 100, 1.5).is_err());
}

#[test]
fn test_lme_report_bootstrap_ci_empty() {
    let mut report = LmeReport::new("m");
    assert!(report.bootstrap_ci("x", &[], 100, 0.05).is_err());
}

#[test]
fn test_lme_report_ab_significance() {
    let mut report = LmeReport::new("model-a");
    report.model_b = Some("model-b".to_string());
    let scores_a = vec![0.8, 0.81, 0.79, 0.82, 0.80];
    let scores_b = vec![0.7, 0.71, 0.69, 0.72, 0.70];
    let p = report
        .ab_significance("bleu", &scores_a, &scores_b, 500)
        .expect("ab sig");
    // Large difference → small p-value
    assert!((0.0..=1.0).contains(&p));
    assert!(p < 0.5); // Should be significant
}

#[test]
fn test_lme_report_ab_no_difference() {
    let mut report = LmeReport::new("m");
    let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let p = report
        .ab_significance("x", &scores, &scores, 200)
        .expect("ab");
    // No difference → high p-value
    assert!(p > 0.3);
}

#[test]
fn test_lme_report_summary() {
    let mut report = LmeReport::new("test-model");
    report.bleu_a = Some(0.42);
    report.rouge_l_a = Some(0.58);
    report.perplexity_a = Some(12.5);
    let s = report.summary();
    assert!(s.contains("test-model"));
    assert!(s.contains("BLEU"));
    assert!(s.contains("ROUGE"));
    assert!(s.contains("Perplexity"));
}

#[test]
fn test_lme_report_ab_empty() {
    let mut report = LmeReport::new("m");
    assert!(report.ab_significance("x", &[], &[1.0], 100).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §A  LmeDistinctN tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_distinct_n_all_unique() {
    let d = LmeDistinctN::new(1).expect("distinct-1");
    let score = d.compute("apple banana cherry date").expect("d1");
    assert!((score - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_distinct_n_all_same() {
    let d = LmeDistinctN::new(1).expect("distinct-1");
    let score = d.compute("cat cat cat cat").expect("d1");
    // Only 1 unique 1-gram out of 4 total
    assert!((score - 0.25).abs() < 1e-10);
}

#[test]
fn test_lme_distinct_n_bigram() {
    let d = LmeDistinctN::new(2).expect("distinct-2");
    let score = d.compute("a b a b a b").expect("d2");
    // Only 1 unique bigram "a b" out of 5 total bigrams
    assert!(score < 0.5);
}

#[test]
fn test_lme_distinct_n_corpus() {
    let d = LmeDistinctN::new(1).expect("distinct-1");
    let texts = ["hello world", "world cup", "hello again"];
    let score = d.compute_corpus(&texts).expect("d1 corpus");
    assert!(score > 0.0 && score <= 1.0);
}

#[test]
fn test_lme_distinct_n_invalid() {
    assert!(LmeDistinctN::new(0).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §B  LmeMeteor tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_meteor_perfect() {
    let m = LmeMeteor::new(0.9, 3.0, 0.5).expect("meteor");
    let score = m
        .compute("the cat sat on the mat", "the cat sat on the mat")
        .expect("meteor");
    assert!(score > 0.9);
}

#[test]
fn test_lme_meteor_no_overlap() {
    let m = LmeMeteor::new(0.9, 3.0, 0.5).expect("meteor");
    let score = m
        .compute("hello world", "goodbye universe")
        .expect("meteor");
    assert!(score < 0.1);
}

#[test]
fn test_lme_meteor_partial() {
    let m = LmeMeteor::new(0.9, 3.0, 0.5).expect("meteor");
    let score = m
        .compute("the cat on the mat", "the cat sat on the mat")
        .expect("meteor");
    assert!(score > 0.0 && score < 1.0);
}

#[test]
fn test_lme_meteor_invalid_alpha() {
    assert!(LmeMeteor::new(0.0, 3.0, 0.5).is_err());
    assert!(LmeMeteor::new(1.5, 3.0, 0.5).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §C  LmeChrF tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_chrf_identical() {
    let c = LmeChrF::new(3, 0, 1.0).expect("chrf");
    let score = c.compute("hello world", "hello world").expect("chrf");
    assert!((score - 1.0).abs() < 1e-8);
}

#[test]
fn test_lme_chrf_no_overlap() {
    let c = LmeChrF::new(2, 0, 1.0).expect("chrf");
    let score = c.compute("xyz", "abc").expect("chrf");
    assert!(score < 0.1);
}

#[test]
fn test_lme_chrf_with_word() {
    let c = LmeChrF::new(3, 1, 1.0).expect("chrf");
    let score = c
        .compute("the cat sat on the mat", "the cat is on the mat")
        .expect("chrf");
    assert!(score > 0.5);
}

#[test]
fn test_lme_chrf_invalid_char_n() {
    assert!(LmeChrF::new(0, 0, 1.0).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §D  LmeCalibration tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_calibration_perfect() {
    // Perfect calibration: conf 0.9 → 90% correct
    let cal = LmeCalibration::new(10).expect("cal");
    let confs: Vec<f64> = vec![0.9; 10];
    let correct: Vec<bool> = vec![true, true, true, true, true, true, true, true, true, false];
    let ece = cal.ece(&confs, &correct).expect("ece");
    assert!(ece < 0.1); // Near-perfect calibration
}

#[test]
fn test_lme_calibration_ece_empty() {
    let cal = LmeCalibration::new(10).expect("cal");
    assert!(cal.ece(&[], &[]).is_err());
}

#[test]
fn test_lme_calibration_mce() {
    let cal = LmeCalibration::new(5).expect("cal");
    let confs = vec![0.9, 0.9, 0.1, 0.1];
    let correct = vec![false, false, true, true];
    let mce = cal.mce(&confs, &correct).expect("mce");
    // Bin at 0.9: conf=0.9, acc=0.0 → gap=0.9; Bin at 0.1: conf=0.1, acc=1.0 → gap=0.9
    assert!(mce > 0.5);
}

#[test]
fn test_lme_calibration_invalid_bins() {
    assert!(LmeCalibration::new(0).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// §E  LmeWER tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_lme_wer_perfect() {
    let wer = LmeWER::new();
    let rate = wer.wer("hello world", "hello world").expect("wer");
    assert!((rate - 0.0).abs() < 1e-10);
}

#[test]
fn test_lme_wer_all_wrong() {
    let wer = LmeWER::new();
    let rate = wer.wer("cat dog", "hello world").expect("wer");
    assert!((rate - 1.0).abs() < 0.01);
}

#[test]
fn test_lme_wer_one_substitution() {
    let wer = LmeWER::new();
    let rate = wer
        .wer("the cat sat on the mat", "the cat sat on the hat")
        .expect("wer");
    // 1 substitution / 6 words = 1/6
    assert!((rate - 1.0 / 6.0).abs() < 1e-10);
}

#[test]
fn test_lme_cer_perfect() {
    let wer = LmeWER::new();
    let rate = wer.cer("hello", "hello").expect("cer");
    assert!((rate - 0.0).abs() < 1e-10);
}

#[test]
fn test_lme_cer_one_edit() {
    let wer = LmeWER::new();
    let rate = wer.cer("helo", "hello").expect("cer");
    // 1 deletion / 5 chars = 0.2
    assert!((rate - 0.2).abs() < 1e-10);
}

#[test]
fn test_lme_wer_batch() {
    let wer = LmeWER::new();
    let hyps = ["the cat", "hello world"];
    let refs = ["the cat", "hello world"];
    let rate = wer.batch_wer(&hyps, &refs).expect("batch wer");
    assert!((rate - 0.0).abs() < 1e-10);
}

#[test]
fn test_lme_wer_batch_mismatch() {
    let wer = LmeWER::new();
    assert!(wer.batch_wer(&["a"], &["b", "c"]).is_err());
}

#[test]
fn test_lme_wer_empty_reference() {
    let wer = LmeWER::new();
    let rate = wer.wer("hello", "").expect("wer empty ref");
    assert!((rate - 1.0).abs() < 1e-10);
}

#[test]
fn test_lme_wer_both_empty() {
    let wer = LmeWER::new();
    let rate = wer.wer("", "").expect("wer both empty");
    assert!((rate - 0.0).abs() < 1e-10);
}
