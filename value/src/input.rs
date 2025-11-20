use bullet::game::{formats::bulletformat::ChessBoard, inputs};
use chess::{Bitboard, Piece, Rays};

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        BASE_INPUTS * 2
    }

    fn max_active(&self) -> usize {
        32
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
        
        let pos = chess::ChessBoard::from(&bbs);

        let (diag, ortho) = pos.generate_pin_masks(pos.side());
        let defender_pin_mask = diag | ortho;

        let (diag, ortho) = pos.generate_pin_masks(pos.side().flipped());
        let attack_pin_mask = diag | ortho;

        let horizontal_mirror = if pos.king_square(pos.side()).file() > 3 {
            7
        } else {
            0
        };

        let occ = pos.occupancy();
        occ.map(|square| {
            let piece = pos.piece_on_square(square);
            let color = pos.color_on_square(square);

            let attack_pin_mask = attack_pin_mask & !Rays::get_ray(square, pos.king_square(pos.side().flipped()));

            let all_attackers = pos.all_attackers_to_square(occ, square);
            let attackers = all_attackers & pos.occupancy_for_side(pos.side().flipped()) & !attack_pin_mask;
            let defenders = all_attackers & pos.occupancy_for_side(pos.side()) & !defender_pin_mask;

            let (attacker, defender) = attacker_defender(&pos, attackers, defenders);

            let piece_index = 64 * (u8::from(piece) - u8::from(Piece::PAWN)) as usize;
            let mut feat = [384, 0][usize::from(color == pos.side())] + piece_index + (usize::from(square) ^ horizontal_mirror);

            if attacker != Piece::NONE {
                feat += 768 * (usize::from(attacker) + 1)
            }

            if defender != Piece::NONE {
                feat += 768 * 7 * (usize::from(defender) + 1)
            }

            f(feat, feat)
        });
    }

    fn shorthand(&self) -> String {
        format!("{}", self.num_inputs())
    }

    fn description(&self) -> String {
        "Extended threat inputs".to_string()
    }
}

const BASE_INPUTS: usize = 18816;

fn attacker_defender(pos: &chess::ChessBoard, attackers: Bitboard, defenders: Bitboard) -> (Piece, Piece) {
    let bb_pawn = pos.piece_mask(Piece::PAWN);
    let bb_knight = pos.piece_mask(Piece::KNIGHT);
    let bb_bishop = pos.piece_mask(Piece::BISHOP);
    let bb_rook = pos.piece_mask(Piece::ROOK);
    let bb_queen = pos.piece_mask(Piece::QUEEN);
    let bb_king = pos.piece_mask(Piece::KING);

    let attacker = if (attackers & bb_pawn).is_not_empty() {
        Piece::PAWN
    } else if (attackers & bb_knight).is_not_empty() {
        Piece::KNIGHT
    } else if (attackers & bb_bishop).is_not_empty() {
        Piece::BISHOP
    } else if (attackers & bb_rook).is_not_empty() {
        Piece::ROOK
    } else if (attackers & bb_queen).is_not_empty() {
        Piece::QUEEN
    } else if (attackers & bb_king).is_not_empty() {
        Piece::KING
    } else {
        Piece::NONE
    };

    let defender = if (defenders & bb_pawn).is_not_empty() {
        Piece::PAWN
    } else if (defenders & bb_knight).is_not_empty() {
        Piece::KNIGHT
    } else if (defenders & bb_bishop).is_not_empty() {
        Piece::BISHOP
    } else if (defenders & bb_rook).is_not_empty() {
        Piece::ROOK
    } else if (defenders & bb_queen).is_not_empty() {
        Piece::QUEEN
    } else if (defenders & bb_king).is_not_empty() {
        Piece::KING
    } else {
        Piece::NONE
    };

    (attacker, defender)
}