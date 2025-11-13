use bullet::game::{formats::{bulletformat::ChessBoard, montyformat::chess::{Piece, Position}}, inputs};

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 * 4
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, board: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in board.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let pos = Position::from_raw(bbs, false, 0, 0, 0, 0);
        let horizontal_mirror = if board.our_ksq() % 8 > 3 {
            7
        } else {
            0
        };

        let threats = pos.threats_by(1);
        let defences = pos.threats_by(0);

        for piece in Piece::PAWN..=Piece::KING {
            let piece_input_index = 64 * (piece - Piece::PAWN) as usize;

            let mut stm_bitboard = pos.piece(piece) & pos.boys();
            let mut nstm_bitboard = pos.piece(piece) & pos.opps();

            while stm_bitboard != 0 {
                let sq = stm_bitboard.trailing_zeros() as usize;
                let mut feat = piece_input_index + (sq ^ horizontal_mirror);

                if threats & (1 << sq) > 0 {
                    feat += 768;
                }

                if defences & (1 << sq) > 0 {
                    feat += 768 * 2;
                }

                f(feat, feat);
                stm_bitboard &= stm_bitboard - 1;
            }

            while nstm_bitboard != 0 {
                let sq = nstm_bitboard.trailing_zeros() as usize;
                let mut feat = 384 + piece_input_index + (sq ^ horizontal_mirror);

                if threats & (1 << sq) > 0 {
                    feat += 768;
                }

                if defences & (1 << sq) > 0 {
                    feat += 768 * 2;
                }

                f(feat, feat);
                nstm_bitboard &= nstm_bitboard - 1;
            }
        }
    }

    fn shorthand(&self) -> String {
        format!("T")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}