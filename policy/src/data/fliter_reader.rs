use std::{
    fs::File,
    io::{BufReader, Read},
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::data::reader::DecompressedData;
use crate::inputs::MAX_MOVES;

#[derive(Clone)]
pub struct FilterDataReader {
    file_path: String,
    buffer_size: usize, // Total samples to hold for shuffling
}

impl FilterDataReader {
    pub fn new(path: &str, buffer_size_mb: usize) -> Self {
        let struct_size = std::mem::size_of::<DecompressedData>();
        let capacity = (buffer_size_mb * 1024 * 1024) / struct_size / 2;
        Self {
            file_path: path.to_string(),
            buffer_size: capacity,
        }
    }

    pub fn map_batches<F: FnMut(&[DecompressedData]) -> bool>(&self, batch_size: usize, mut f: F) {
        let file_path = self.file_path.clone();
        let buffer_size = self.buffer_size;
        let struct_size = std::mem::size_of::<DecompressedData>();

        // 1. Thread: Read raw structs from disk into sample batches
        let (raw_sender, raw_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(8);
        std::thread::spawn(move || {
            loop {
                let file = File::open(&file_path).expect("Failed to open training file");
                let mut reader = BufReader::new(file);
                
                // We read in chunks to keep the pipeline moving
                let chunk_count = 16384; 
                
                loop {
                    let mut chunk = Vec::with_capacity(chunk_count);
                    unsafe {
                        chunk.set_len(chunk_count);
                        let slice = std::slice::from_raw_parts_mut(
                            chunk.as_mut_ptr() as *mut u8,
                            chunk_count * struct_size
                        );
                        
                        if reader.read_exact(slice).is_err() {
                            break; // EOF or Error
                        }
                    }
                    if raw_sender.send(chunk).is_err() { return; }
                }
                println!("Data reader reached EOF, restarting stream...");
            }
        });

        // 2. Thread: Shuffle Buffer
        let (shuffled_sender, shuffled_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(2);
        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);
            
            while let Ok(chunk) = raw_receiver.recv() {
                shuffle_buffer.extend(chunk);
                
                if shuffle_buffer.len() >= buffer_size {
                    shuffle(&mut shuffle_buffer);
                    if shuffled_sender.send(shuffle_buffer).is_err() { return; }
                    shuffle_buffer = Vec::with_capacity(buffer_size);
                }
            }
        });

        // 3. Main Loop: Consume Shuffled Batches
        'training: while let Ok(inputs) = shuffled_receiver.recv() {
            for batch in inputs.chunks(batch_size) {
                if f(batch) {
                    break 'training;
                }
            }
        }
    }
}

// --- UTILS ---

fn shuffle(data: &mut [DecompressedData]) {
    let mut rng = Rand::with_seed();
    for i in (1..data.len()).rev() {
        let idx = (rng.rng() as usize) % (i + 1);
        data.swap(idx, i);
    }
}

pub struct Rand(u64);
impl Rand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
        Self(seed)
    }
    pub fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}