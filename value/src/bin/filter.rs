use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::sync::{Arc, Mutex};

// IMPORTS: Adjust these paths to match your actual crate structure
use bullet::{
    montyformat::chess::{Move, Position},
    game::formats::bulletformat::ChessBoard, // Assuming this is the target output format
};

use montyformat::{
    FastDeserialise, MontyValueFormat,
    chess::{Move, Position},
};

// --- CONFIGURATION ---
const INPUT_PATH: &str = "interleaved-value.bin";
const OUTPUT_PATH: &str = "finetune-data.bin";
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
                    process_game(&game_bytes, &mut local_buffer, |pos, mv, score, result| {
                        // --- FILTER LOGIC ---
                        true 
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