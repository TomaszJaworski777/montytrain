use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write, Error, ErrorKind};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use montyformat::chess::{Attacks, Castling, Flag, Move, Piece, Position, Side};
use montyformat::{FastDeserialise, MontyFormat, SearchData};

const INPUT_PATH: &str = "interleaved-policy.bin";
const OUTPUT_PATH: &str = "finetune-policy.bin";
const THREADS: usize = 6;
const BATCH_SIZE: usize = 1024;
const MAX_MOVES: usize = 64; 

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DecompressedData {
    pub pos: Position,
    pub castling: Castling,
    pub moves: [(u16, u16); MAX_MOVES],
    pub num: usize,
}

fn main() -> std::io::Result<()> {
    println!("--- Policy Sacrifice Filter Starting ---");
    println!("Source: {}\nTarget: {}", INPUT_PATH, OUTPUT_PATH);

    let (work_sender, work_receiver) = sync_channel::<Vec<Vec<u8>>>(THREADS * 4);
    let (write_sender, write_receiver) = sync_channel::<Vec<DecompressedData>>(THREADS * 4);

    // 1. Writer Thread
    let writer_handle = thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::new(File::create(OUTPUT_PATH)?);
        let mut count = 0usize;
        
        while let Ok(batch) = write_receiver.recv() {
            for data in batch {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        &data as *const DecompressedData as *const u8,
                        std::mem::size_of::<DecompressedData>(),
                    )
                };
                writer.write_all(bytes)?;
                count += 1;
            }
        }
        writer.flush()?;
        println!("\nWriter finished. Total aggressive policy samples: {}", count);
        Ok(())
    });

    // 2. Worker Threads
    let work_receiver = Arc::new(Mutex::new(work_receiver));
    let mut worker_handles = Vec::new();

    for _ in 0..THREADS {
        let rx = work_receiver.clone();
        let tx = write_sender.clone();

        worker_handles.push(thread::spawn(move || {
            loop {
                let batch = {
                    let lock = rx.lock().unwrap();
                    match lock.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    }
                };

                let mut local_buffer = Vec::new();

                for game_bytes in batch {
                    process_policy_game(&game_bytes, &mut local_buffer);
                }

                if !local_buffer.is_empty() && tx.send(local_buffer).is_err() {
                    break;
                }
            }
        }));
    }

    drop(write_sender);

    // 3. Reader Loop
    let input_file = File::open(INPUT_PATH)?;
    let total_size = input_file.metadata()?.len();
    let mut reader = BufReader::new(input_file);
    
    let mut current_batch = Vec::with_capacity(BATCH_SIZE);
    let mut bytes_read = 0u64;
    let start_time = Instant::now();
    let mut last_print = Instant::now();

    loop {
        let mut buffer = Vec::new();
        // Uses your FastDeserialise implementation to chunk games
        if MontyFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_err() || buffer.is_empty() {
            break;
        }

        bytes_read += buffer.len() as u64;
        current_batch.push(buffer);

        if current_batch.len() >= BATCH_SIZE {
            work_sender.send(current_batch).unwrap();
            current_batch = Vec::with_capacity(BATCH_SIZE);

            if last_print.elapsed().as_millis() > 500 {
                print_progress(bytes_read, total_size, start_time);
                last_print = Instant::now();
            }
        }
    }

    if !current_batch.is_empty() { work_sender.send(current_batch).unwrap(); }
    print_progress(bytes_read, total_size, start_time);
    drop(work_sender);

    for h in worker_handles { h.join().unwrap(); }
    writer_handle.join().unwrap()?;

    println!("\nFiltering Complete.");
    Ok(())
}

fn process_policy_game(game_bytes: &[u8], output: &mut Vec<DecompressedData>) {
    let mut reader = Cursor::new(game_bytes);

    if let Ok(game) = MontyFormat::deserialise_from(&mut reader) {
        let length = game.moves.len();
        let mut pos = game.startpos;
        let castling = game.castling;
        let result = game.result; // 1.0 = White Win, 0.0 = Black Win, 0.5 = Draw

        for data in game.moves {
            // Check if this specific board + move is "Aggressive & Winning"
            if is_fast_win(&pos, &castling, &data, result, length) || 
                is_aggressive_win(&pos, &castling, &data, result, data.best_move) ||
                is_attacking_win(&pos, &castling, &data, result, data.best_move) {
                let mut moves_array = [(0u16, 0u16); MAX_MOVES];
                let mut num: usize = 0;

                if let Some(ref dist) = data.visit_distribution {
                    for (i, &(m, visits)) in dist.iter().enumerate().take(MAX_MOVES) {
                        moves_array[i] = (u16::from(m), visits as u16);
                        num += 1;
                    }
                } else {
                    // Fallback to best move if distribution is missing
                    moves_array[0] = (u16::from(data.best_move), 100);
                    num = 1;
                }

                output.push(DecompressedData {
                    pos: pos.clone(),
                    castling: castling.clone(),
                    moves: moves_array,
                    num,
                });
            }
            pos.make(data.best_move, &castling);
        }
    }
}

fn is_aggressive_win(pos: &Position, castling: &Castling, data: &SearchData, game_result: f32, best_move: Move) -> bool {
    if pos.stm() == 0 && game_result < 0.9 || pos.stm() == 1 && game_result > 0.1 {
        return false;
    }

    if best_move.flag() == Flag::KS || best_move.flag() == Flag::QS {
        return false;
    }

    let material_balance = calculate_material(pos);
    if pos.piece(Piece::QUEEN).count_ones() > 2 || material_balance.abs() > 1000 {
        return false;
    }

    let filter = |pos: &Position, result: f32, material: i32| -> bool {
        let white_sac = pos.stm() == 0 && result > 0.9 && material < -300 && material > -2000;
        let black_sac = pos.stm() == 1 && result < 0.1 && material < -300 && material > -2000;

        white_sac || black_sac
    };

    if !filter(pos, game_result, material_balance) {
        return false;
    }

    let mut pos = Position::from_raw(pos.bbs(), pos.stm() == 1, pos.enp_sq(), 0, pos.halfm(), pos.fullm());
    let material = ab(&pos, &castling, -30000, 30000, 0);

    let mut castling = Castling::default();
    if !filter(&pos, game_result, material) {
        return false;
    }

    return !see(&pos, &best_move, -300);
}

fn is_attacking_win(pos: &Position, castling: &Castling, data: &SearchData, game_result: f32, best_move: Move) -> bool {
    if pos.stm() == 0 && game_result < 0.9 || pos.stm() == 1 && game_result > 0.1 {
        return false;
    }

    let material_balance = calculate_material(pos);
    if pos.piece(Piece::QUEEN).count_ones() > 2 || material_balance.abs() > 1000 {
        return false;
    }

    let king_sq = pos.king_sq(1 - pos.stm());
    let king_rank = (king_sq / 8) as i32;
    let king_file = (king_sq % 8) as i32;

    let best_move_rank = (best_move.to() / 8) as i32;
    let best_move_file = (best_move.to() % 8) as i32;

    let distance = (king_file - best_move_file).abs() + (king_rank - best_move_rank).abs();

    if (pos.stm() == 0 && best_move_rank < 4) || (pos.stm() == 1 && best_move_rank > 3) {
        return false;
    }

    return distance <= 4;
}

fn is_fast_win(pos: &Position, castling: &Castling, data: &SearchData, game_result: f32, game_length: usize) -> bool {
    if pos.stm() == 0 && game_result < 0.9 || pos.stm() == 1 && game_result > 0.1 {
        return false;
    }

    let material_balance = calculate_material(pos);
    if pos.piece(Piece::QUEEN).count_ones() > 2 || material_balance.abs() > 1000 {
        return false;
    }

    let filter = |pos: &Position, result: f32, material: i32| -> bool {
        let white_sac = pos.stm() == 0 && result > 0.9 && material < -300 && material > -2000;
        let black_sac = pos.stm() == 1 && result < 0.1 && material < -300 && material > -2000;

        white_sac || black_sac
    };

    if !filter(pos, game_result, material_balance) {
        return false;
    }

    let mut pos = Position::from_raw(pos.bbs(), pos.stm() == 1, pos.enp_sq(), 0, pos.halfm(), pos.fullm());
    let material = ab(&pos, &castling, -30000, 30000, 0);

    let mut castling = Castling::default();
    if !filter(&pos, game_result, material) {
        return false;
    }

    return game_length <= 40;
}

fn print_progress(bytes_read: u64, total_bytes: u64, start_time: Instant) {
    let percentage = (bytes_read as f64 / total_bytes as f64) * 100.0;
    let elapsed = start_time.elapsed().as_secs_f64();
    let speed = (bytes_read as f64 / 1_048_576.0) / elapsed;
    print!("\rProgress: {:5.2}% | Speed: {:6.2} MB/s | Filtered for Policy Finetuning...", percentage, speed);
    let _ = std::io::stdout().flush();
}

fn ab(pos: &Position, castling: &Castling, mut alpha: i32, beta: i32, depth: u8) -> i32 {
    let in_check = pos.in_check();
    if !in_check {
        let eval = calculate_material(pos);
        if eval >= beta { return beta; }
        if eval > alpha { alpha = eval; }
    }
    if depth > 4 { return if in_check { alpha } else { calculate_material(pos) }; }

    let mut move_list = Vec::new();
    pos.map_legal_moves(castling, |mv| {
        move_list.push(mv);
    });

    for mv in move_list {
        let mut pos_cpy = pos.clone();
        pos_cpy.make(mv, castling);
        let score = -ab(&pos_cpy, castling, -beta, -alpha, depth + 1);
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}

#[inline]
fn calculate_material(pos: &Position) -> i32 {
    let stm_mask = pos.boys(); 
    let nstm_mask = pos.opps();
    let mut score = 0;
    let pieces = [
        (Piece::PAWN, 100), (Piece::KNIGHT, 300), (Piece::BISHOP, 325),
        (Piece::ROOK, 500), (Piece::QUEEN, 900)
    ];
    for (p, val) in pieces {
        score += (pos.piece(p) & stm_mask).count_ones() as i32 * val;
        score -= (pos.piece(p) & nstm_mask).count_ones() as i32 * val;
    }
    score
}

#[inline]
fn mv_is_check(mv: Move, pos: &Position, castling: &Castling) -> bool {
    let mut pos_clone = pos.clone();
    pos_clone.make(mv, castling);
    pos_clone.in_check()
}

pub fn see(pos: &Position, mov: &Move, threshold: i32) -> bool {
    let sq = usize::from(mov.to());
    assert!(sq < 64, "wha");
    let mut next = if mov.is_promo() {
        mov.promo_pc()
    } else {
        pos.get_pc(1 << mov.src())
    };
    let mut score = gain(pos, mov) - threshold - SEE_VALS[next];

    if score >= 0 {
        return true;
    }

    let mut occ = (pos.bbs()[Side::WHITE] | pos.bbs()[Side::BLACK]) ^ (1 << mov.src()) ^ (1 << sq);
    if mov.is_en_passant() {
        occ ^= 1 << (sq ^ 8);
    }

    let bishops = pos.bbs()[Piece::BISHOP] | pos.bbs()[Piece::QUEEN];
    let rooks = pos.bbs()[Piece::ROOK] | pos.bbs()[Piece::QUEEN];
    let mut us = 1 - pos.stm();
    let mut attackers = (Attacks::knight(sq) & pos.bbs()[Piece::KNIGHT])
        | (Attacks::king(sq) & pos.bbs()[Piece::KING])
        | (Attacks::pawn(sq, Side::WHITE) & pos.bbs()[Piece::PAWN] & pos.bbs()[Side::BLACK])
        | (Attacks::pawn(sq, Side::BLACK) & pos.bbs()[Piece::PAWN] & pos.bbs()[Side::WHITE])
        | (Attacks::rook(sq, occ) & rooks)
        | (Attacks::bishop(sq, occ) & bishops);

    loop {
        let our_attackers = attackers & pos.bbs()[us];
        if our_attackers == 0 {
            break;
        }

        for pc in Piece::PAWN..=Piece::KING {
            let board = our_attackers & pos.bbs()[pc];
            if board > 0 {
                occ ^= board & board.wrapping_neg();
                next = pc;
                break;
            }
        }

        if [Piece::PAWN, Piece::BISHOP, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::bishop(sq, occ) & bishops;
        }
        if [Piece::ROOK, Piece::QUEEN].contains(&next) {
            attackers |= Attacks::rook(sq, occ) & rooks;
        }

        attackers &= occ;
        score = -score - 1 - SEE_VALS[next];
        us ^= 1;

        if score >= 0 {
            if next == Piece::KING && attackers & pos.bbs()[us] > 0 {
                us ^= 1;
            }
            break;
        }
    }

    (pos.stm() == 1) != (us == 1)
}

fn gain(pos: &Position, mov: &Move) -> i32 {
    if mov.is_en_passant() {
        return SEE_VALS[Piece::PAWN];
    }
    let mut score = SEE_VALS[pos.get_pc(1 << mov.to())];
    if mov.is_promo() {
        score += SEE_VALS[mov.promo_pc()] - SEE_VALS[Piece::PAWN];
    }
    score
}

pub const SEE_VALS: [i32; 8] = [0, 0, 100, 450, 450, 650, 1250, 0];