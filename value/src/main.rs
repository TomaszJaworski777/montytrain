mod arch;
mod input;
mod threads_extended;

use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use arch::make_trainer;
use input::ThreatInputs;

use bullet::{
    game::formats::{bulletformat::ChessBoard, montyformat::chess::{Attacks, Castling, Piece}}, nn::optimiser, trainer::{
        default::{
            formats::montyformat::chess::{Move, Position},
            loader,
        },
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    }
};

const HIDDEN_SIZE: usize = 4096;
const END_SUPERBATCH: usize = 25;

pub const QA: i16 = 128;
pub const QB: i16 = 1024;

fn main() {
    let mut trainer = make_trainer::<ThreatInputs>(HIDDEN_SIZE);

    let size = 2 * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2;

    let schedule = TrainingSchedule {
        net_id: format!("MontyThreatsFT2-{size}"),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 65_536,
            batches_per_superbatch: 1526,
            start_superbatch: 1,
            end_superbatch: END_SUPERBATCH,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: /*0.001,*/ 0.0000005,
            final_lr: /*0.0000001,*/ 0.000000005,
            final_superbatch: END_SUPERBATCH,
        },
        save_rate: 5,
    };

    let optimiser_params = optimiser::AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    trainer.optimiser.set_params(optimiser_params);

    let settings = LocalSettings {
        threads: 8,
        test_set: None,
        output_directory: "value_checkpoints",
        batch_queue_size: 32,
    };

    fn filter(pos: &Position, mv: Move, _: i16, result: f32) -> bool {
        if calculate_material(pos) > -300 {
            return false;
        }

        let pos = &Position::from_raw(pos.bbs(), pos.stm() == 1, pos.enp_sq(), 0, pos.halfm(), pos.fullm());

        if pos.piece(Piece::QUEEN).count_ones() > 2 
            || pos.piece(Piece::ROOK).count_ones() > 5 
            || pos.piece(Piece::KNIGHT).count_ones() > 5 
            || pos.piece(Piece::BISHOP).count_ones() > 5 
        {
            return false;
        }

        let fen = pos.as_fen();
        let mut castling = Castling::default();
        castling.parse(pos, &fen);

        let qs_score = qsearch(pos, &castling, -30000, 30000, 0);

        let white_sac = pos.stm() == 0 && result > 0.9 && qs_score < -300 && qs_score > -2000;
        let black_sac = pos.stm() == 1 && result < 0.1 && qs_score < -300 && qs_score > -2000;

        let filter = white_sac || black_sac;

        // if filter {
        //     println!("passed with result {result}, qsearch {}: {}", qs_score, fen)
        // } 
        // else {
        //     println!("not passed {}, result: {}", pos.as_fen(), result)
        // }

        filter
    }

    let data_loader = loader::MontyBinpackLoader::new(
        "./interleaved-value.bin",
        2,
        1,
        filter,
    );

    trainer.load_from_checkpoint("value_checkpoints/MontyThreats-4000");
    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "8/kpp5/b7/8/2Q5/8/8/5K2 w - - 0 1",
        "1k1rqb2/2ppprpp/pp3p2/8/2P5/8/1QPPPPPP/RR4K1 w - - 0 1",
    ] {
        let vals = trainer.eval_raw_output(fen);
        println!("FEN: {fen}");
        
        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                let total = win + draw + loss;
                
                println!("EVAL: ({}, {}, {})", win / total, draw / total, loss / total)
            }
            _ => panic!("Invalid output size!"),
        }
    }
}

fn qsearch(pos: &Position, castling: &Castling, mut alpha: i32, beta: i32, depth: u8) -> i32 {
    let in_check = pos.in_check();

    if !in_check {
        let eval = calculate_material(pos);

        if eval >= beta {
            return beta;
        }

        if eval > alpha {
            alpha = eval;
        }
    }

    if depth > 2 {
        return if in_check { alpha } else { calculate_material(pos) };
    }

    let mut move_list = Vec::new();
    pos.map_legal_moves(castling, |mv| {
        if in_check || mv.is_capture() || mv.is_promo() || mv_is_check(mv, pos, castling) {
            move_list.push(mv)
        }
    });
    move_list.sort_by(|a, b| get_move_value(pos, *b).cmp(&get_move_value(pos, *a)));

    for (idx, &mv) in move_list.iter().enumerate() {
        if mv == Move::NULL {
            continue;
        }

        let mut pos_cpy = pos.clone();
        pos_cpy.make(mv, castling);

        let score = -qsearch(&pos_cpy, castling, -beta, -alpha, depth + 1);

        if score >= beta {
            return beta;
        }

        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

#[inline]
fn get_move_value(pos: &Position, mv: Move) -> i32 {
    let mut result: i32 = 0;

    if mv.is_capture() {
        let moving_piece = pos.get_pc(1 << mv.src());
        let target_piece = pos.get_pc(1 << mv.to());
        result += ((target_piece + 1) as i32 * 100) - (moving_piece + 1) as i32;
    }

    if mv.is_promo() {
        result += ((mv.promo_pc() + 1) as i32) * 100;
    }

    return result;
}

fn calculate_material(pos: &Position) -> i32 {
    const PIECE_VALUES: [i32; 7] = [0, 0, 100, 300, 300, 500, 900];
    let mut result = 0;

    let mut occ = pos.boys();

    for _ in 0..=1 {
        for piece in Piece::PAWN..=Piece::QUEEN {
            let piece_mask = pos.piece(piece) & occ;
            result += piece_mask.count_ones() as i32 * PIECE_VALUES[piece as usize];
        }
        result = -result;
        occ = pos.opps();
    }

    result
}

fn mv_is_check(mv: Move, pos: &Position, castling: &Castling) -> bool {
    let mut pos_clone = pos.clone();
    pos_clone.make(mv, castling);
    pos_clone.in_check()
}
