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
    buffer_size: usize,
    threads: usize,
}

impl FilterDataReader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        let struct_size = std::mem::size_of::<DecompressedData>();
        let capacity = (buffer_size_mb * 1024 * 1024) / struct_size;
        Self {
            file_path: path.to_string(),
            buffer_size: capacity,
            threads,
        }
    }

    pub fn map_batches<F: FnMut(&[DecompressedData]) -> bool>(&self, batch_size: usize, mut f: F) {
        let file_path = self.file_path.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let struct_size = std::mem::size_of::<DecompressedData>();
        
        let samples_per_thread = 1024;
        let total_samples_per_batch = samples_per_thread * threads;

        let (raw_sender, raw_receiver) = mpsc::sync_channel::<Vec<u8>>(8);

        std::thread::spawn(move || {
            loop {
                let file = File::open(&file_path).expect("Failed to open training file");
                let mut reader = BufReader::new(file);
                let bytes_per_batch = total_samples_per_batch * struct_size;

                loop {
                    let mut buffer = vec![0u8; bytes_per_batch];
                    if reader.read_exact(&mut buffer).is_err() {
                        break; // EOF
                    }
                    if raw_sender.send(buffer).is_err() { return; }
                }
                println!("Reader: EOF reached, looping file.");
            }
        });

        let (struct_sender, struct_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(threads);

        std::thread::spawn(move || {
            while let Ok(raw_batch) = raw_receiver.recv() {
                let sender = struct_sender.clone();
                
                std::thread::scope(|s| {
                    for chunk in raw_batch.chunks(samples_per_thread * struct_size) {
                        let thread_sender = sender.clone();
                        s.spawn(move || {
                            let mut samples = Vec::with_capacity(samples_per_thread);
                            
                            for struct_bytes in chunk.chunks_exact(struct_size) {
                                let sample: DecompressedData = unsafe {
                                    std::ptr::read(struct_bytes.as_ptr() as *const _)
                                };
                                
                                if sample.num > MAX_MOVES {
                                    panic!("CORRUPT DATA: num {} exceeds MAX_MOVES.", sample.num);
                                }
                                
                                samples.push(sample);
                            }
                            let _ = thread_sender.send(samples);
                        });
                    }
                });
            }
        });

        let (shuffled_sender, shuffled_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(2);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);
            while let Ok(batch) = struct_receiver.recv() {
                shuffle_buffer.extend(batch);
                
                if shuffle_buffer.len() >= buffer_size {
                    shuffle(&mut shuffle_buffer);
                    if shuffled_sender.send(shuffle_buffer).is_err() { return; }
                    shuffle_buffer = Vec::with_capacity(buffer_size);
                }
            }
        });

        'training: while let Ok(inputs) = shuffled_receiver.recv() {
            for batch in inputs.chunks(batch_size) {
                if f(batch) {
                    break 'training;
                }
            }
        }
    }
}

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