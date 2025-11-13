mod arch;
mod input;

use arch::make_trainer;
use input::ThreatInputs;

use bullet::{
    nn::optimiser,
    trainer::{
        default::{
            formats::montyformat::chess::{Move, Position},
            loader,
        },
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
};

const HIDDEN_SIZE: usize = 2048;
const END_SUPERBATCH: usize = 800;

pub const QA: i16 = 128;
pub const QB: i16 = 1024;

fn main() {
    let mut trainer = make_trainer::<ThreatInputs>(HIDDEN_SIZE);

    let schedule = TrainingSchedule {
        net_id: "800sb".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: END_SUPERBATCH,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.00001,
            final_superbatch: END_SUPERBATCH,
        },
        save_rate: 10,
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
        threads: 4,
        test_set: None,
        output_directory: "value_checkpoints",
        batch_queue_size: 32,
    };

    fn filter(_: &Position, _: Move, _: i16, _: f32) -> bool {
        true
    }

    let data_loader = loader::MontyBinpackLoader::new(
        "./interleaved-value.bin",
        128000,
        8,
        filter,
    );

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
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
