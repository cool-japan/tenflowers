use super::*;


// --- CodeTokenizer tests ---

#[test]
fn test_tokenizer_python_keywords() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("def foo(x):", Language::Python);
    let kinds: Vec<&TokenKind> = tokens.iter().map(|t| &t.kind).collect();
    assert!(
        kinds.contains(&&TokenKind::Keyword),
        "def should be a keyword"
    );
    assert!(
        kinds.contains(&&TokenKind::Identifier),
        "foo/x should be identifiers"
    );
}

#[test]
fn test_tokenizer_rust_keywords() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("fn main() { let x = 42; }", Language::Rust);
    let kw_texts: Vec<&str> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::Keyword)
        .map(|t| t.text.as_str())
        .collect();
    assert!(kw_texts.contains(&"fn"), "fn is a Rust keyword");
    assert!(kw_texts.contains(&"let"), "let is a Rust keyword");
}

#[test]
fn test_tokenizer_c_keywords() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("int main() { return 0; }", Language::C);
    let kw_texts: Vec<&str> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::Keyword)
        .map(|t| t.text.as_str())
        .collect();
    assert!(kw_texts.contains(&"int"), "int is a C keyword");
    assert!(kw_texts.contains(&"return"), "return is a C keyword");
}

#[test]
fn test_tokenizer_line_comment_python() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("x = 1 # comment here", Language::Python);
    assert!(tokens.iter().any(|t| t.kind == TokenKind::Comment));
}

#[test]
fn test_tokenizer_block_comment() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("/* block comment */ int x;", Language::C);
    assert!(tokens.iter().any(|t| t.kind == TokenKind::Comment));
}

#[test]
fn test_tokenizer_string_literal() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("x = \"hello world\"", Language::Generic);
    assert!(tokens
        .iter()
        .any(|t| t.kind == TokenKind::Literal && t.text.contains("hello")));
}

#[test]
fn test_tokenizer_numeric_literal() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("x = 3.14", Language::Generic);
    assert!(tokens
        .iter()
        .any(|t| t.kind == TokenKind::Literal && t.text == "3.14"));
}

#[test]
fn test_tokenizer_operators() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("a == b && c != d", Language::Generic);
    let ops: Vec<&str> = tokens
        .iter()
        .filter(|t| t.kind == TokenKind::Operator)
        .map(|t| t.text.as_str())
        .collect();
    assert!(ops.contains(&"=="));
    assert!(ops.contains(&"&&"));
    assert!(ops.contains(&"!="));
}

// --- ASTEncoder tests ---

#[test]
fn test_ast_encoder_single_node() {
    let encoder = ASTEncoder::new(16);
    let node = AstNode::new("FunctionDef", 0, 0);
    let enc = encoder.encode_node(&node, 16);
    assert_eq!(enc.len(), 16);
}

#[test]
fn test_ast_encoder_depth_differs() {
    let encoder = ASTEncoder::new(16);
    let node0 = AstNode::new("Stmt", 0, 0);
    let node1 = AstNode::new("Stmt", 3, 0);
    let enc0 = encoder.encode_node(&node0, 16);
    let enc1 = encoder.encode_node(&node1, 16);
    assert_ne!(
        enc0, enc1,
        "Different depths should produce different encodings"
    );
}

#[test]
fn test_ast_encoder_tree_bfs_order() {
    let encoder = ASTEncoder::new(8);
    let mut root = AstNode::new("Root", 0, 0);
    root.add_child(AstNode::new("ChildA", 1, 0));
    root.add_child(AstNode::new("ChildB", 1, 1));
    let encodings = encoder.encode_tree(&root);
    assert_eq!(encodings.len(), 3, "Root + 2 children = 3 nodes");
    assert_eq!(encodings[0].len(), 8);
}

#[test]
fn test_ast_encoder_empty_tree() {
    let encoder = ASTEncoder::new(8);
    let leaf = AstNode::new("Leaf", 2, 1);
    let encodings = encoder.encode_tree(&leaf);
    assert_eq!(encodings.len(), 1);
}

// --- CodeBert tests ---

#[test]
fn test_codebert_forward_output_shape() {
    let config = CodeBertConfig::new(100, 16, 2, 2, 32);
    let model = CodeBert::new(config);
    let tokens = vec![1usize, 5, 3, 7];
    let hidden = model.forward(&tokens);
    assert_eq!(hidden.len(), 4);
    assert_eq!(hidden[0].len(), 16);
}

#[test]
fn test_codebert_mlm_logits_shape() {
    let config = CodeBertConfig::new(50, 16, 2, 1, 32);
    let model = CodeBert::new(config);
    let tokens = vec![0usize, 1, 2];
    let hidden = model.forward(&tokens);
    let logits = model.mlm_logits(&hidden);
    assert_eq!(logits.len(), 3);
    assert_eq!(logits[0].len(), 50);
}

#[test]
fn test_codebert_seq_longer_than_max() {
    let config = CodeBertConfig::new(50, 8, 2, 1, 4);
    let model = CodeBert::new(config);
    let tokens: Vec<usize> = (0..10).collect();
    let hidden = model.forward(&tokens);
    // Should be capped at max_seq_len
    assert_eq!(hidden.len(), 4);
}

// --- CodeContrastive tests ---

#[test]
fn test_code_contrastive_encode_shapes() {
    let cc = CodeContrastive::new(100, 16);
    let code_tokens = vec![1usize, 2, 3, 4];
    let doc_tokens = vec![5usize, 6, 7];
    let ce = cc.encode_code(&code_tokens);
    let de = cc.encode_doc(&doc_tokens);
    assert_eq!(ce.len(), 16);
    assert_eq!(de.len(), 16);
}

#[test]
fn test_code_contrastive_l2_normalized() {
    let cc = CodeContrastive::new(100, 16);
    let tokens = vec![1usize, 2, 3];
    let emb = cc.encode_code(&tokens);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "embedding should be unit-norm, got {}",
        norm
    );
}

#[test]
fn test_contrastive_loss_positive() {
    let cc = CodeContrastive::new(100, 8);
    let code_embeds = vec![vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let doc_embeds = vec![vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let loss = cc.contrastive_loss(&code_embeds, &doc_embeds, 0.07);
    assert!(loss >= 0.0, "Loss should be non-negative");
}

#[test]
fn test_contrastive_loss_batch_n2() {
    let cc = CodeContrastive::new(50, 8);
    let code_embeds: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let doc_embeds: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let loss = cc.contrastive_loss(&code_embeds, &doc_embeds, 0.1);
    assert!(loss.is_finite());
}

// --- FlashFillSolver tests ---

#[test]
fn test_flashfill_upper() {
    let solver = FlashFillSolver::new();
    let examples = [("hello", "HELLO")];
    let prog = solver.synthesize(&examples);
    assert!(prog.is_some(), "Should find uppercase program");
    let prog = prog.expect("should find uppercase program");
    assert_eq!(FlashFillSolver::execute(&prog, "world"), "WORLD");
}

#[test]
fn test_flashfill_lower() {
    let solver = FlashFillSolver::new();
    let examples = [("HELLO", "hello")];
    let prog = solver.synthesize(&examples);
    assert!(prog.is_some());
    let prog = prog.expect("should find lowercase program");
    assert_eq!(FlashFillSolver::execute(&prog, "WORLD"), "world");
}

#[test]
fn test_flashfill_strip() {
    let solver = FlashFillSolver::new();
    let examples = [("  hello  ", "hello")];
    let prog = solver.synthesize(&examples);
    assert!(prog.is_some());
    let prog = prog.expect("should find strip program");
    assert_eq!(FlashFillSolver::execute(&prog, "  rust  "), "rust");
}

#[test]
fn test_flashfill_split_first_word() {
    let solver = FlashFillSolver::new();
    let examples = [("John Smith", "John")];
    let prog = solver.synthesize(&examples);
    assert!(prog.is_some());
    let prog = prog.expect("should find split-first-word program");
    let out = FlashFillSolver::execute(&prog, "Alice Brown");
    assert_eq!(out, "Alice");
}

#[test]
fn test_flashfill_execute_replace() {
    let prog = FlashFillProgram {
        ops: vec![StringDsl::Replace("-".to_string(), "_".to_string())],
    };
    assert_eq!(
        FlashFillSolver::execute(&prog, "foo-bar-baz"),
        "foo_bar_baz"
    );
}

#[test]
fn test_flashfill_execute_substr() {
    let prog = FlashFillProgram {
        ops: vec![StringDsl::Substr(0, 2)],
    };
    assert_eq!(FlashFillSolver::execute(&prog, "hello"), "hel");
}

// --- NeuralProgramInducer tests ---

#[test]
fn test_differentiable_interpreter_add() {
    let prog = vec![Instruction {
        op: OpCode::Add,
        args: vec![0, 1, 2],
    }];
    let mut interp = DifferentiableInterpreter::new(4, prog);
    let regs = interp.execute_soft(&[3.0, 4.0]);
    assert!(
        (regs[2] - 7.0).abs() < 1e-4,
        "register[2] should be 7.0, got {}",
        regs[2]
    );
}

#[test]
fn test_differentiable_interpreter_copy() {
    let prog = vec![Instruction {
        op: OpCode::Copy,
        args: vec![0, 0, 3],
    }];
    let mut interp = DifferentiableInterpreter::new(4, prog);
    let regs = interp.execute_soft(&[42.0]);
    assert!((regs[3] - 42.0).abs() < 1e-4);
}

#[test]
fn test_differentiable_interpreter_div_zero() {
    let prog = vec![Instruction {
        op: OpCode::Div,
        args: vec![0, 1, 2],
    }];
    let mut interp = DifferentiableInterpreter::new(4, prog);
    let regs = interp.execute_soft(&[10.0, 0.0]);
    // Should not panic; result is 0.0 (safe division)
    assert_eq!(regs[2], 0.0);
}

#[test]
fn test_differentiable_interpreter_soft_and() {
    let prog = vec![Instruction {
        op: OpCode::And,
        args: vec![0, 1, 2],
    }];
    let mut interp = DifferentiableInterpreter::new(4, prog);
    let regs = interp.execute_soft(&[10.0, 10.0]);
    // Both sigmas ≈ 1, so AND ≈ 1
    assert!(
        regs[2] > 0.9,
        "soft AND of large positive values should be ~1, got {}",
        regs[2]
    );
}

// --- CodeSummarizer tests ---

#[test]
fn test_code_summarizer_encode_source() {
    let config = PointerGeneratorConfig {
        vocab_size: 50,
        hidden_dim: 16,
        attn_dim: 8,
    };
    let cs = CodeSummarizer::new(config);
    let tokens = vec![1usize, 2, 3, 4, 5];
    let states = cs.encode_source(&tokens);
    assert_eq!(states.len(), 5);
    assert_eq!(states[0].len(), 16);
}

#[test]
fn test_code_summarizer_decode_step() {
    let config = PointerGeneratorConfig {
        vocab_size: 50,
        hidden_dim: 16,
        attn_dim: 8,
    };
    let cs = CodeSummarizer::new(config);
    let tokens = vec![1usize, 2, 3];
    let states = cs.encode_source(&tokens);
    let hidden = vec![0.0f32; 16];
    let (vocab_dist, copy_dist) = cs.decode_step(0, &hidden, &states);
    assert_eq!(vocab_dist.len(), 50);
    assert_eq!(copy_dist.len(), 3);
    let vocab_sum: f32 = vocab_dist.iter().sum();
    assert!(
        (vocab_sum - 1.0).abs() < 1e-4,
        "vocab dist should sum to 1, got {}",
        vocab_sum
    );
}

#[test]
fn test_code_summarizer_copy_mechanism() {
    let config = PointerGeneratorConfig {
        vocab_size: 50,
        hidden_dim: 16,
        attn_dim: 8,
    };
    let cs = CodeSummarizer::new(config);
    let attn_weights = vec![0.5f32, 0.3, 0.2];
    let src_tokens = vec![1usize, 2, 3];
    let copy_vocab = cs.copy_mechanism(&attn_weights, &src_tokens, 50);
    assert_eq!(copy_vocab.len(), 50);
    let total: f32 = copy_vocab.iter().sum();
    assert!((total - 1.0).abs() < 1e-4, "copy vocab sums to {}", total);
}

// --- BugLocalizerGnn tests ---

#[test]
fn test_bug_localizer_forward_shape() {
    let gnn = BugLocalizerGnn::new(50, 8, 2);
    let cfg = ControlFlowGraph {
        nodes: vec![
            CfgNode {
                tokens: vec![1, 2, 3],
                successors: vec![1],
            },
            CfgNode {
                tokens: vec![4, 5],
                successors: vec![2],
            },
            CfgNode {
                tokens: vec![6],
                successors: vec![],
            },
        ],
    };
    let scores = gnn.forward(&cfg);
    assert_eq!(scores.len(), 3);
    for &s in &scores {
        assert!((0.0..=1.0).contains(&s), "score should be in [0,1], got {}", s);
    }
}

#[test]
fn test_bug_localizer_top_k() {
    let gnn = BugLocalizerGnn::new(50, 8, 2);
    let scores = vec![0.1, 0.9, 0.5, 0.3];
    let top2 = gnn.top_k_suspicious(&scores, 2);
    assert_eq!(top2.len(), 2);
    assert_eq!(top2[0].0, 1, "node 1 has highest score");
    assert!((top2[0].1 - 0.9).abs() < 1e-6);
}

#[test]
fn test_bug_localizer_empty_cfg() {
    let gnn = BugLocalizerGnn::new(50, 8, 2);
    let cfg = ControlFlowGraph { nodes: vec![] };
    let scores = gnn.forward(&cfg);
    assert!(scores.is_empty());
}

// --- TestCaseGenerator tests ---

#[test]
fn test_boundary_tests_count() {
    let bounds = vec![(0.0f32, 1.0), (0.0, 10.0)];
    let tests = TestCaseGenerator::generate_boundary_tests(2, &bounds);
    assert_eq!(tests.len(), 4, "2 inputs => 2^2 = 4 corners");
}

#[test]
fn test_boundary_tests_values() {
    let bounds = vec![(0.0f32, 1.0)];
    let tests = TestCaseGenerator::generate_boundary_tests(1, &bounds);
    assert_eq!(tests.len(), 2);
    let vals: Vec<f32> = tests.iter().map(|t| t.input[0]).collect();
    assert!(vals.contains(&0.0));
    assert!(vals.contains(&1.0));
}

#[test]
fn test_mutate_test_negate() {
    let tc = TestCase {
        input: vec![1.0, 2.0, 3.0],
        expected_output: vec![6.0],
    };
    let mut rng = StdRng::seed_from_u64(1);
    let mutated =
        TestCaseGenerator::mutate_test(&tc, MutationOperator::NegateCondition, &mut rng);
    // At least one value should differ
    let changed = tc
        .input
        .iter()
        .zip(mutated.input.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6)
        || tc
            .expected_output
            .iter()
            .zip(mutated.expected_output.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "NegateCondition should mutate some element");
}

#[test]
fn test_mutate_test_off_by_one() {
    let tc = TestCase {
        input: vec![5.0],
        expected_output: vec![5.0],
    };
    let mut rng = StdRng::seed_from_u64(2);
    let mutated = TestCaseGenerator::mutate_test(&tc, MutationOperator::OffByOne, &mut rng);
    assert!((mutated.input[0] - 5.0).abs() < 2.0 + 1e-6);
    assert!(
        (mutated.input[0] - 5.0).abs() > 1e-6,
        "OffByOne should change the value by 1"
    );
}

#[test]
fn test_mutate_test_null_input() {
    let tc = TestCase {
        input: vec![3.0, 4.0],
        expected_output: vec![7.0],
    };
    let mut rng = StdRng::seed_from_u64(3);
    let mutated = TestCaseGenerator::mutate_test(&tc, MutationOperator::NullInput, &mut rng);
    assert!(
        mutated.input.contains(&0.0),
        "NullInput should zero out one element"
    );
}

// --- CodeMetrics tests ---

#[test]
fn test_cyclomatic_complexity_linear() {
    // A linear program: E = N-1, P = 1 => V = (N-1) - N + 2 = 1
    let v = CodeMetrics::cyclomatic_complexity(4, 5, 1);
    assert_eq!(v, 1);
}

#[test]
fn test_cyclomatic_complexity_branching() {
    // Single if-else: adds 1 edge => V = E - N + 2P = 6-5+2 = 3
    let v = CodeMetrics::cyclomatic_complexity(6, 5, 1);
    assert_eq!(v, 3);
}

#[test]
fn test_halstead_metrics() {
    // Small example: 3 operators, 4 operands, 2 unique ops, 3 unique operands
    let m = CodeMetrics::halstead_metrics(3, 4, 2, 3);
    assert_eq!(m.vocabulary, 5);
    assert_eq!(m.length, 7);
    assert!(m.volume > 0.0);
    assert!(m.difficulty > 0.0);
    assert!(m.effort > 0.0);
}

#[test]
fn test_halstead_zero_operands() {
    // Edge case: unique_operands = 0 => difficulty = 0
    let m = CodeMetrics::halstead_metrics(2, 0, 2, 0);
    assert_eq!(m.difficulty, 0.0);
}

#[test]
fn test_maintainability_index_range() {
    let mi = CodeMetrics::maintainability_index(1000.0, 5.0, 100);
    assert!(
        (0.0..=100.0).contains(&mi),
        "MI should be in [0, 100], got {}",
        mi
    );
}

#[test]
fn test_maintainability_index_small_program() {
    // Very small program => high MI
    let mi = CodeMetrics::maintainability_index(10.0, 1.0, 5);
    assert!(
        mi > 50.0,
        "Small simple program should have high MI, got {}",
        mi
    );
}

#[test]
fn test_count_sloc() {
    let source = "// This is a comment\nfn main() {\n    let x = 1;\n\n    let y = 2;\n}\n";
    let sloc = CodeMetrics::count_sloc(source);
    assert_eq!(
        sloc, 4,
        "Expected 4 non-blank non-comment lines, got {}",
        sloc
    );
}

#[test]
fn test_count_halstead_tokens() {
    let tokenizer = CodeTokenizer::new();
    let tokens = tokenizer.tokenize("x = x + 1", Language::Generic);
    let (n_ops, n_operands, unique_ops, unique_operands) =
        CodeMetrics::count_halstead_tokens(&tokens);
    assert!(n_ops > 0, "should have at least one operator/keyword");
    assert!(n_operands > 0, "should have at least one operand");
    assert!(unique_ops <= n_ops);
    assert!(unique_operands <= n_operands);
}

#[test]
fn test_codebert_forward_no_nan() {
    let config = CodeBertConfig::new(50, 8, 2, 1, 16);
    let model = CodeBert::new(config);
    let tokens = vec![0usize, 5, 10, 15, 20];
    let hidden = model.forward(&tokens);
    for row in &hidden {
        for &v in row {
            assert!(
                v.is_finite(),
                "Expected finite value in hidden state, got {}",
                v
            );
        }
    }
}

#[test]
fn test_flashfill_replace_hyphen_underscore() {
    let solver = FlashFillSolver::new();
    let examples = [("foo-bar", "foo_bar"), ("baz-qux", "baz_qux")];
    let prog = solver.synthesize(&examples);
    assert!(prog.is_some(), "Should find a replace program");
}

#[test]
fn test_differentiable_interpreter_min_max() {
    let prog = vec![
        Instruction {
            op: OpCode::Max,
            args: vec![0, 1, 2],
        },
        Instruction {
            op: OpCode::Min,
            args: vec![0, 1, 3],
        },
    ];
    let mut interp = DifferentiableInterpreter::new(5, prog);
    let regs = interp.execute_soft(&[3.0, 7.0]);
    assert!((regs[2] - 7.0).abs() < 1e-4, "Max(3, 7) should be 7");
    assert!((regs[3] - 3.0).abs() < 1e-4, "Min(3, 7) should be 3");
}

// ---------------------------------------------------------------------------
// neural_exec: NpiController tests
// ---------------------------------------------------------------------------

#[test]
fn test_program_library_add_primitive() {
    use scirs2_core::random::SeedableRng;
    use scirs2_core::random::rngs::StdRng;
    let mut lib = ProgramLibrary::new(8);
    let mut rng = StdRng::seed_from_u64(1);
    lib.add_primitive("add", &mut rng);
    lib.add_primitive("sub", &mut rng);
    assert_eq!(lib.entries.len(), 2);
    assert!(lib.get("add").is_some());
    assert!(lib.get("sub").is_some());
    assert!(lib.get("mul").is_none());
}

#[test]
fn test_program_library_nearest() {
    use scirs2_core::random::SeedableRng;
    use scirs2_core::random::rngs::StdRng;
    let mut lib = ProgramLibrary::new(4);
    let mut rng = StdRng::seed_from_u64(2);
    lib.add_primitive("p0", &mut rng);
    lib.add_primitive("p1", &mut rng);
    // Query with the embedding of p0
    let query = lib.entries[0].embedding.clone();
    let idx = lib.nearest(&query);
    assert_eq!(idx, Some(0), "nearest should find exact match");
}

#[test]
fn test_program_library_learned() {
    let mut lib = ProgramLibrary::new(4);
    let emb = vec![1.0f32, 0.0, 0.0, 0.0];
    lib.add_learned("learned_prog", emb.clone());
    let entry = lib.get("learned_prog").expect("should find learned program");
    assert_eq!(entry.embedding, emb);
    assert!(!entry.is_primitive);
}

#[test]
fn test_npi_controller_lstm_step() {
    let ctrl = NpiController::new(8, 2, 42);
    let x = vec![0.0f32; 8];
    let h = vec![0.0f32; 8];
    let c = vec![0.0f32; 8];
    let (new_h, new_c) = ctrl.lstm_step(&x, &h, &c);
    assert_eq!(new_h.len(), 8);
    assert_eq!(new_c.len(), 8);
    for &v in &new_h {
        assert!(v.is_finite(), "hidden state should be finite");
    }
}

#[test]
fn test_npi_controller_predict_terminate() {
    let ctrl = NpiController::new(8, 2, 99);
    let h = vec![0.0f32; 8];
    let p = ctrl.predict_terminate(&h);
    assert!((0.0..=1.0).contains(&p), "termination prob in [0,1], got {}", p);
}

#[test]
fn test_npi_controller_run_episode() {
    let ctrl = NpiController::new(8, 2, 7);
    let env = vec![0.5f32; 8];
    let prog = vec![0.1f32; 8];
    let steps = ctrl.run_episode(&env, &prog, 5);
    assert!(!steps.is_empty(), "should have at least one step");
    assert!(steps.len() <= 5, "should not exceed max_steps");
}

#[test]
fn test_recursive_npi_execute() {
    let npi = RecursiveNpi::new(8, 2, 2, 11);
    let env = vec![0.0f32; 8];
    let prog = vec![0.2f32; 8];
    let calls = npi.execute(&env, &prog, 0);
    // Should terminate without panicking
    assert!(calls.len() < 1000, "recursive calls should be bounded");
}

// ---------------------------------------------------------------------------
// neural_exec: TapeLanguage and DifferentiableVm tests
// ---------------------------------------------------------------------------

#[test]
fn test_tape_language_execute_hard_increment() {
    let tape = TapeLanguage::new(8, 10);
    let prog = vec![TapeOp::Increment, TapeOp::Output];
    let (_, output) = tape.execute_hard(&prog, &[0]);
    assert_eq!(output, vec![1]);
}

#[test]
fn test_tape_language_execute_hard_move() {
    let tape = TapeLanguage::new(8, 10);
    let prog = vec![TapeOp::Right, TapeOp::Increment, TapeOp::Output];
    let (final_tape, _) = tape.execute_hard(&prog, &[0; 8]);
    assert_eq!(final_tape[1], 1, "cell 1 should be incremented");
}

#[test]
fn test_tape_language_execute_hard_decrement() {
    let tape = TapeLanguage::new(8, 10);
    let prog = vec![TapeOp::Increment, TapeOp::Increment, TapeOp::Decrement, TapeOp::Output];
    let (_, output) = tape.execute_hard(&prog, &[0]);
    assert_eq!(output, vec![1]);
}

#[test]
fn test_tape_language_execute_soft_no_nan() {
    let tape = TapeLanguage::new(4, 6);
    let uniform = vec![0.125f32; 8];
    let op_probs = vec![uniform.clone(); 4];
    let result = tape.execute_soft(&op_probs, &[0.0; 4]);
    for v in &result {
        assert!(v.is_finite(), "soft tape values should be finite");
    }
}

#[test]
fn test_tape_op_index_roundtrip() {
    for op in TapeOp::all() {
        let idx = op.index();
        assert!(idx < 8, "op index should be < 8");
        assert_eq!(TapeOp::all()[idx], op, "index should roundtrip");
    }
}

#[test]
fn test_differentiable_vm_forward_soft() {
    let vm = DifferentiableVm::new(4, 4, 8, 42);
    let tape = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = vm.forward_soft(&tape);
    assert_eq!(output.len(), 4);
    for v in &output {
        assert!(v.is_finite());
    }
}

#[test]
fn test_differentiable_vm_sample_program() {
    use scirs2_core::random::SeedableRng;
    use scirs2_core::random::rngs::StdRng;
    let vm = DifferentiableVm::new(6, 4, 8, 55);
    let mut rng = StdRng::seed_from_u64(1);
    let prog = vm.sample_program(&mut rng, 1.0);
    assert_eq!(prog.len(), 6);
}

#[test]
fn test_beam_search_returns_program() {
    let searcher = ProgramSearchBeam::new(3, 4, 0.6);
    let prog = searcher.search(|_ops| {
        // Uniform distribution over ops
        vec![0.125f32; 8]
    });
    assert!(prog.is_some(), "beam search should return a program");
    let p = prog.expect("should have program");
    assert_eq!(p.len(), 4, "program length should equal max_len");
}

// ---------------------------------------------------------------------------
// neural_exec: Type-Guided Synthesis tests
// ---------------------------------------------------------------------------

#[test]
fn test_simple_type_unify_int() {
    let mut subst = Substitution::new();
    assert!(unify(&SimpleType::Int, &SimpleType::Int, &mut subst));
}

#[test]
fn test_simple_type_unify_var() {
    let mut subst = Substitution::new();
    let t = SimpleType::Var(0);
    assert!(unify(&t, &SimpleType::Int, &mut subst));
    assert_eq!(subst.apply(&t), SimpleType::Int);
}

#[test]
fn test_simple_type_unify_fail() {
    let mut subst = Substitution::new();
    assert!(!unify(&SimpleType::Int, &SimpleType::Bool, &mut subst));
}

#[test]
fn test_typed_dsl_eval_add() {
    let expr = TypedDslExpr::Add(
        Box::new(TypedDslExpr::IntLit(3)),
        Box::new(TypedDslExpr::IntLit(4)),
    );
    let val = TypedDsl::eval(&expr);
    assert_eq!(val, Some(DslValue::Int(7)));
}

#[test]
fn test_typed_dsl_eval_neg() {
    let expr = TypedDslExpr::Neg(Box::new(TypedDslExpr::IntLit(5)));
    let val = TypedDsl::eval(&expr);
    assert_eq!(val, Some(DslValue::Int(-5)));
}

#[test]
fn test_typed_dsl_eval_and() {
    let expr = TypedDslExpr::And(
        Box::new(TypedDslExpr::BoolLit(true)),
        Box::new(TypedDslExpr::BoolLit(false)),
    );
    let val = TypedDsl::eval(&expr);
    assert_eq!(val, Some(DslValue::Bool(false)));
}

#[test]
fn test_typed_dsl_eval_eq() {
    let expr = TypedDslExpr::Eq(
        Box::new(TypedDslExpr::IntLit(2)),
        Box::new(TypedDslExpr::IntLit(2)),
    );
    let val = TypedDsl::eval(&expr);
    assert_eq!(val, Some(DslValue::Bool(true)));
}

#[test]
fn test_typed_dsl_type_error() {
    // Bool + Int is ill-typed
    let expr = TypedDslExpr::Add(
        Box::new(TypedDslExpr::BoolLit(true)),
        Box::new(TypedDslExpr::IntLit(1)),
    );
    assert!(TypedDsl::infer_type(&expr).is_none());
    assert!(TypedDsl::eval(&expr).is_none());
}

#[test]
fn test_type_guided_enumerator_int() {
    let enumerator = TypeGuidedEnumerator::new(2, vec![vec![0i64], vec![1]]);
    let exprs = enumerator.enumerate(&SimpleType::Int);
    assert!(!exprs.is_empty(), "should find at least some Int expressions");
    for e in &exprs {
        assert_eq!(TypedDsl::infer_type(e), Some(SimpleType::Int));
    }
}

#[test]
fn test_type_guided_enumerator_bool() {
    let enumerator = TypeGuidedEnumerator::new(2, vec![vec![0i64]]);
    let exprs = enumerator.enumerate(&SimpleType::Bool);
    assert!(!exprs.is_empty(), "should find Bool expressions");
}

#[test]
fn test_observational_equivalence_dedup() {
    let equiv = ObservationalEquivalence::new(vec![vec![0i64], vec![1]]);
    let exprs = vec![
        TypedDslExpr::IntLit(0),
        TypedDslExpr::IntLit(0), // duplicate
        TypedDslExpr::IntLit(1),
    ];
    let unique = equiv.deduplicate(exprs);
    assert_eq!(unique.len(), 2, "should deduplicate identical literals");
}

// ---------------------------------------------------------------------------
// neural_exec: IoEmbedder and SyntaxGuidedSearch tests
// ---------------------------------------------------------------------------

#[test]
fn test_io_embedder_shape() {
    let embedder = IoEmbedder::new(8, 4, 16, 42);
    let examples = vec![
        ExampleIo::new(vec![1.0, 2.0, 0.0, 0.0], vec![3.0, 0.0, 0.0, 0.0]),
        ExampleIo::new(vec![0.5, 0.5, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]),
    ];
    let emb = embedder.encode(&examples);
    assert_eq!(emb.len(), 16);
    for v in &emb {
        assert!(v.is_finite());
    }
}

#[test]
fn test_io_embedder_empty() {
    let embedder = IoEmbedder::new(8, 4, 16, 1);
    let emb = embedder.encode(&[]);
    assert_eq!(emb.len(), 16);
    assert!(emb.iter().all(|&v| v == 0.0));
}

#[test]
fn test_syntax_guided_search_best_derivation() {
    let embedder = IoEmbedder::new(4, 2, 8, 1);
    let rules = vec![
        PcfgRule {
            lhs: "E".to_string(),
            rhs: vec!["x".to_string()],
            log_prob: -0.5,
        },
        PcfgRule {
            lhs: "E".to_string(),
            rhs: vec!["1".to_string()],
            log_prob: -1.0,
        },
    ];
    let searcher = SyntaxGuidedSearch::new(rules, "E".to_string(), 3, embedder);
    let best = searcher.best_derivation();
    assert!(best.is_some(), "should find a derivation");
    assert_eq!(best.expect("derivation"), vec!["x"], "x has higher prob");
}

#[test]
fn test_syntax_guided_search_rules_for() {
    let embedder = IoEmbedder::new(4, 2, 8, 1);
    let rules = vec![
        PcfgRule {
            lhs: "E".to_string(),
            rhs: vec!["a".to_string()],
            log_prob: -1.0,
        },
        PcfgRule {
            lhs: "T".to_string(),
            rhs: vec!["b".to_string()],
            log_prob: -1.0,
        },
    ];
    let searcher = SyntaxGuidedSearch::new(rules, "E".to_string(), 2, embedder);
    assert_eq!(searcher.rules_for("E").len(), 1);
    assert_eq!(searcher.rules_for("T").len(), 1);
    assert_eq!(searcher.rules_for("X").len(), 0);
}

// ---------------------------------------------------------------------------
// neural_exec: PsMetrics tests
// ---------------------------------------------------------------------------

#[test]
fn test_ps_metrics_empty() {
    let m = PsMetrics::new();
    assert_eq!(m.exact_match_rate(), 0.0);
    assert_eq!(m.generalization_rate(), 0.0);
    assert_eq!(m.mean_time_ms(), 0.0);
}

#[test]
fn test_ps_metrics_record() {
    let mut m = PsMetrics::new();
    m.record(true, true, 10.0);
    m.record(false, true, 20.0);
    m.record(true, false, 5.0);
    assert_eq!(m.n_total, 3);
    assert_eq!(m.n_exact, 2);
    assert_eq!(m.n_generalized, 2);
    assert!((m.exact_match_rate() - 2.0 / 3.0).abs() < 1e-6);
    assert!((m.mean_time_ms() - 35.0 / 3.0).abs() < 1e-3);
}
