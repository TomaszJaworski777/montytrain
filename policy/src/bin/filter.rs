use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write, Error, ErrorKind};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use montyformat::chess::{Castling, Flag, Move, Piece, Position};
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
        let mut pos = game.startpos;
        let castling = game.castling;
        let result = game.result; // 1.0 = White Win, 0.0 = Black Win, 0.5 = Draw

        for data in game.moves {
            // Check if this specific board + move is "Aggressive & Winning"
            if is_aggressive_win(&pos, &castling, &data, result, data.best_move) {
                let mut moves_array = [(0u16, 0u16); MAX_MOVES];
                let mut num = 0;

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

                println!("{}, {}", pos.as_fen(), data.best_move);

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
    let old_v = ab(&pos, &castling, -30000, 30000, 0);

    let mut castling = Castling::default();
    if !filter(&pos, game_result, old_v) {
        return false;
    }

    pos.make(best_move, &castling);
    let new_v = -ab(&pos, &castling, -30000, 30000, 0);

    if new_v.abs() > 10000 {
        return false;
    }

    if new_v + 300 <= old_v {
        println!("{new_v} < {old_v}");
        return true;
    }
    return false;
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
        // if mv.is_capture() || mv.is_promo() || (depth == 0 && mv_is_check(mv, pos, castling)) {
        //     move_list.push(mv);
        // }

        if mv.is_capture() && mv_is_check(mv, pos, castling) {
            move_list.push(mv);
        }
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