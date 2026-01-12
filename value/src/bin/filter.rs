use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// IMPORTS: Adjust these paths to match your actual crate structure
use bullet::{
    game::formats::montyformat::{
        FastDeserialise, MontyValueFormat,
        chess::{Move, Position},
    },
    game::formats::bulletformat::ChessBoard, // Assuming this is the target output format
};

// --- CONFIGURATION ---
const INPUT_PATH: &str = "interleaved-value.bin";
const OUTPUT_PATH: &str = "finetune-value.bin";
const THREADS: usize = 6;
const BATCH_SIZE: usize = 1024; // Increased slightly for smoother progress updates

fn main() -> std::io::Result<()> {
    println!("Starting conversion: {} -> {}", INPUT_PATH, OUTPUT_PATH);

    // 1. Setup Channels
    let (work_sender, work_receiver) = sync_channel::<Vec<Vec<u8>>>(THREADS * 4);
    let (write_sender, write_receiver) = sync_channel::<Vec<u8>>(THREADS * 4);

    // 2. Spawn Writer Thread
    let writer_handle = thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::new(File::create(OUTPUT_PATH)?);
        let mut count = 0usize;
        
        while let Ok(data) = write_receiver.recv() {
            if !data.is_empty() {
                writer.write_all(&data)?;
                count += data.len() / std::mem::size_of::<ChessBoard>();
            }
        }
        
        writer.flush()?;
        println!("\nWriter finished. Total positions written: {}", count);
        Ok(())
    });

    // 3. Spawn Worker Threads
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
                    process_game(&game_bytes, &mut local_buffer, |pos, _, _, result| {
                        if pos.piece(Piece::QUEEN).count_ones() > 2 
                            || pos.piece(Piece::ROOK).count_ones() > 5 
                            || pos.piece(Piece::KNIGHT).count_ones() > 5 
                            || pos.piece(Piece::BISHOP).count_ones() > 5 
                        {
                            return false;
                        }

                        if calculate_material(pos) > -300 {
                            return false;
                        }

                        let pos = &Position::from_raw(pos.bbs(), pos.stm() == 1, pos.enp_sq(), 0, pos.halfm(), pos.fullm());

                        let mut castling = Castling::default();
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
                    });
                }

                if tx.send(local_buffer).is_err() {
                    break;
                }
            }
        }));
    }

    drop(write_sender);

    // 4. Reader Loop (Main Thread) with Progress
    let input_file = File::open(INPUT_PATH)?;
    let total_size = input_file.metadata()?.len(); // For progress calculation
    let mut reader = BufReader::new(input_file);
    
    let mut current_batch = Vec::with_capacity(BATCH_SIZE);
    let mut bytes_read = 0u64;
    let start_time = Instant::now();
    let mut last_print = Instant::now();

    loop {
        let mut buffer = Vec::new();
        
        if MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer).is_err() {
            break;
        }

        if buffer.is_empty() {
            break;
        }

        // Track progress
        bytes_read += buffer.len() as u64;
        current_batch.push(buffer);

        // Send batch if full
        if current_batch.len() >= BATCH_SIZE {
            work_sender.send(current_batch).unwrap();
            current_batch = Vec::with_capacity(BATCH_SIZE);

            // Update UI roughly every 500ms to avoid console spam
            if last_print.elapsed().as_millis() > 500 {
                print_progress(bytes_read, total_size, start_time);
                last_print = Instant::now();
            }
        }
    }

    // Send remainder
    if !current_batch.is_empty() {
        work_sender.send(current_batch).unwrap();
    }
    
    // Final progress update (100%)
    print_progress(bytes_read, total_size, start_time);

    drop(work_sender);

    // 5. Cleanup
    for h in worker_handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap()?;

    println!("\nDone.");
    Ok(())
}

fn print_progress(bytes_read: u64, total_bytes: u64, start_time: Instant) {
    let percentage = (bytes_read as f64 / total_bytes as f64) * 100.0;
    
    let elapsed_secs = start_time.elapsed().as_secs_f64();
    let mb_read = bytes_read as f64 / 1_048_576.0;
    let speed = if elapsed_secs > 0.0 { mb_read / elapsed_secs } else { 0.0 };

    // \r returns cursor to start of line, allowing us to overwrite it
    print!(
        "\rProgress: {:5.2}% | {:7.2} MB read | Speed: {:6.2} MB/s", 
        percentage, mb_read, speed
    );
    std::io::stdout().flush().unwrap();
}

fn process_game<F>(game_bytes: &[u8], output_buffer: &mut Vec<u8>, filter: F)
where
    F: Fn(&Position, Move, i16, f32) -> bool,
{
    let mut reader = Cursor::new(game_bytes);

    if let Ok(game) = MontyValueFormat::deserialise_from(&mut reader, Vec::new()) {
        let mut pos = game.startpos;
        let castling = game.castling;

        for data in game.moves {
            if filter(&pos, data.best_move, data.score, game.result) {
                if let Ok(board) = ChessBoard::from_raw(pos.bbs(), pos.stm(), data.score, game.result) {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &board as *const ChessBoard as *const u8,
                            std::mem::size_of::<ChessBoard>(),
                        )
                    };
                    output_buffer.extend_from_slice(bytes);
                }
            }
            pos.make(data.best_move, &castling);
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

    if depth > 6 {
        return if in_check { alpha } else { calculate_material(pos) };
    }

    let mut move_list = Vec::new();
    pos.map_legal_moves(castling, |mv| {
        if depth == 0 && (mv.is_promo() || mv_is_check(mv, pos, castling)) {
            move_list.push((mv, get_move_value(pos, mv)));
            return;
        }

        if depth == 1 && in_check {
            move_list.push((mv, get_move_value(pos, mv)));
            return;
        }

        if mv.is_capture() {
            move_list.push((mv, get_move_value(pos, mv)))
        }
    });

    move_list.sort_by(|(_, a), (_, b)| b.cmp(&a));

    for (idx, &(mv, _)) in move_list.iter().enumerate() {
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

#[inline]
fn calculate_material(pos: &Position) -> i32 {
    let stm = pos.boys(); 
    let nstm = pos.opps();

    let mut score = 0;
    score += (pos.piece(Piece::PAWN) & stm).count_ones() as i32 * 100;
    score += (pos.piece(Piece::KNIGHT) & stm).count_ones() as i32 * 300;
    score += (pos.piece(Piece::BISHOP) & stm).count_ones() as i32 * 300;
    score += (pos.piece(Piece::ROOK) & stm).count_ones() as i32 * 500;
    score += (pos.piece(Piece::QUEEN) & stm).count_ones() as i32 * 900;

    score -= (pos.piece(Piece::PAWN) & nstm).count_ones() as i32 * 100;
    score -= (pos.piece(Piece::KNIGHT) & nstm).count_ones() as i32 * 300;
    score -= (pos.piece(Piece::BISHOP) & nstm).count_ones() as i32 * 300;
    score -= (pos.piece(Piece::ROOK) & nstm).count_ones() as i32 * 500;
    score -= (pos.piece(Piece::QUEEN) & nstm).count_ones() as i32 * 900;

    score
}

#[inline]
fn mv_is_check(mv: Move, pos: &Position, castling: &Castling) -> bool {
    let mut pos_clone = pos.clone();
    pos_clone.make(mv, castling);
    pos_clone.in_check()
}
