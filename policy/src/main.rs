pub mod data;
pub mod inputs;
pub mod model;

use acyclib::{
    device::Device,
    trainer::{
        optimiser::{
            adam::{AdamW, AdamWParams},
            Optimiser,
        },
        schedule::{TrainingSchedule, TrainingSteps},
        Trainer,
    },
};
use bullet_cuda_backend::CudaDevice;

use data::MontyDataLoader;

const NAME: &str = "policy-8192";
const HL_SIZE: usize = 8192;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 20;

// const START_LR: f32 = 0.001;
// const END_LR: f32 = 0.00001; 

const START_LR: f32 = 0.000005;
const END_LR: f32 = 0.00000005; 

fn preamble() {
    println!("NAME:             {NAME}");
    println!("HL:               {HL_SIZE}");
    println!("START_SUPERBATCH: {START_SUPERBATCH}");
    println!("END_SUPERBATCH:   {END_SUPERBATCH}");
    println!("HSTART_LRL:       {START_LR}");
    println!("END_LR:           {END_LR}");
}

fn main() {
    let dataloader = MontyDataLoader::new("./finetune-policy.bin", 16000, 4, 4);

    let device = CudaDevice::new(0).unwrap();

    let (graph, node) = model::make(device, HL_SIZE);

    let params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -0.99, max_weight: 0.99 };
    let optimiser = Optimiser::<_, _, AdamW<_>>::new(graph, params).unwrap();

    let mut trainer = Trainer { optimiser, state: () };

    let save_rate = 5;

    let steps = TrainingSteps { batch_size: 16384, batches_per_superbatch: 6104, start_superbatch: START_SUPERBATCH, end_superbatch: END_SUPERBATCH };

    let schedule = TrainingSchedule {
        steps,
        log_rate: 64,
        lr_schedule: Box::new(|_, sb| {
            if sb >= END_SUPERBATCH {
                return END_LR;
            }

            let lambda = sb as f32 / END_SUPERBATCH as f32;
            START_LR * (END_LR / START_LR).powf(lambda)
        }),
    };

    trainer.optimiser.load_from_checkpoint("./policy_checkpoints/policy-8192-800");

    preamble();
    trainer
        .train_custom(
            schedule,
            dataloader,
            |_, _, _, _| {},
            |trainer, superbatch| {
                if superbatch % save_rate == 0 || superbatch == steps.end_superbatch {
                    println!("Saving Checkpoint");
                    let dir = format!("./policy_checkpoints/{NAME}-{superbatch}");
                    let _ = std::fs::create_dir(&dir);
                    trainer.optimiser.write_to_checkpoint(&dir).unwrap();
                    model::save_quantised(&trainer.optimiser.graph, &format!("{dir}/quantised.bin")).unwrap();
                    //model::save_raw(&trainer.optimiser.graph, &format!("{dir}/raw.bin")).unwrap();
                }
            },
        )
        .unwrap();

    model::eval(&mut trainer.optimiser.graph, node, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    model::eval(&mut trainer.optimiser.graph, node, "rk6/8/8/p7/P7/Q7/R7/RK6 w - - 80 200");
    model::eval(&mut trainer.optimiser.graph, node, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    model::eval(&mut trainer.optimiser.graph, node, "8/8/p5p1/2bk1p1p/5P1P/1P3PK1/8/4B3 b - - 3 48");
}

