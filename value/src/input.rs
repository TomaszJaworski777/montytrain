use bullet::game::{formats::bulletformat::ChessBoard, inputs};
use chess::Bitboard;

use crate::threads_extended::ThreatsExtended;

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        ThreatsExtended::INPUT_SIZE
    }

    fn max_active(&self) -> usize {
        256
    }

    fn map_features<F: FnMut(usize, usize)>(&self, board: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [Bitboard::EMPTY; 8];
        for (pc, sq) in board.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let board = &chess::ChessBoard::from(&bbs);
        ThreatsExtended::map_inputs(board, |feat| f(feat, feat));
    }

    fn shorthand(&self) -> String {
        format!("{}", self.num_inputs())
    }

    fn description(&self) -> String {
        "Extended threat inputs".to_string()
    }
}