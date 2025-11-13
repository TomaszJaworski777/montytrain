use bullet::{
    game::inputs::SparseInputType,
    nn::{
        optimiser::{AdamW, AdamWOptimiser}, Shape,
    },
    trainer::save::SavedFormat,
    value::{NoOutputBuckets, ValueTrainer, ValueTrainerBuilder},
};

use crate::{QA, QB};

pub fn make_trainer<T: Default + SparseInputType>(
    l1: usize,
) -> ValueTrainer<AdamWOptimiser, T, NoOutputBuckets> {
    let inputs = T::default();
    let num_inputs = inputs.num_inputs();

    ValueTrainerBuilder::default()
        .wdl_output()
        .inputs(T::default())
        .optimiser(AdamW)
        .save_format(&[
            SavedFormat::id("l0w").quantise::<i8>(QA).round(),
            SavedFormat::id("l0b").quantise::<i8>(QA).round(),
            SavedFormat::id("l1w").quantise::<i16>(QB).round(),
            SavedFormat::id("l1b").quantise::<i16>(QB).round(),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build_custom(|builder, inputs, targets| {
            let l0 = builder.new_affine("l0", num_inputs, l1);
            let l1 = builder.new_affine("l1", l1, 16);
            let l2 = builder.new_affine("l2", 16, 128);
            let l3 = builder.new_affine("l3", 128, 3);

            l0.init_with_effective_input_size(32);

            let l0 = l0.forward(inputs).screlu();
            let l1 = l1.forward(l0).screlu();
            let l2 = l2.forward(l1).screlu();
            let out = l3.forward(l2);

            let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
            let loss = ones.matmul(out.softmax_crossentropy_loss(targets));

            (out, loss)
        })
}
