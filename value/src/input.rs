use bullet::game::{formats::bulletformat::ChessBoard, inputs};
use chess::{Bitboard, Piece, Rays, Side, Square};

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

    fn map_features<F: FnMut(usize, usize)>(&self, board: &Self::RequiredDataType, f: F) {
        let mut pos = chess::ChessBoard::default();

        for (pc, sq) in board.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);

            pos.set_piece_on_square(Square::from(sq), Piece::from(pt - 2), Side::from(c as u8));
        }

        base_inputs(&pos, f);
    }

    fn shorthand(&self) -> String {
        format!("T")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}

const BASE_INPUTS: usize = 18816;

fn base_inputs<F: FnMut(usize, usize)>(board: &chess::ChessBoard, mut f: F) {
    let horizontal_mirror = if board.king_square(board.side()).file() > 3 {
        7
    } else {
        0
    };

    let occ = board.occupancy();
    occ.map(|square| {
        let piece = board.piece_on_square(square);
        let color = board.color_on_square(square);

        let (attacker, defender) = attacker_defender(board, square, occ);

        let piece_index = 64 * (u8::from(piece) - u8::from(Piece::PAWN)) as usize;
        let mut feat = [384, 0][usize::from(color == board.side())] + piece_index + (usize::from(square) ^ horizontal_mirror);

        if attacker != Piece::NONE {
            feat += 768 * (usize::from(attacker) + 1)
        }

        if defender != Piece::NONE {
            feat += 768 * 7 * (usize::from(defender) + 1)
        }

        f(feat, feat)
    });
}


fn attacker_defender(board: &chess::ChessBoard, square: Square, occ: Bitboard) -> (Piece, Piece) {
    let (diag, ortho) = board.generate_pin_masks(board.side());
    let defender_pin_mask = diag | ortho;

    let (diag, ortho) = board.generate_pin_masks(board.side().flipped());
    let attack_pin_mask = (diag | ortho) & !Rays::get_ray(square, board.king_square(board.side().flipped()));

    let all_attackers = board.all_attackers_to_square(occ, square);
    let attackers = all_attackers & board.occupancy_for_side(board.side().flipped()) & !attack_pin_mask;
    let defenders = all_attackers & board.occupancy_for_side(board.side()) & !defender_pin_mask;

    let piece_types = [
        Piece::PAWN,
        Piece::KNIGHT,
        Piece::BISHOP,
        Piece::ROOK,
        Piece::QUEEN,
        Piece::KING
    ];

    let mut attacker = Piece::NONE;
    let mut defender = Piece::NONE;

    for &piece in piece_types.iter() {
        let piece_mask = board.piece_mask(piece);

        if attacker == Piece::NONE && (attackers & piece_mask).is_not_empty() {
            attacker = piece;
        }

        if defender == Piece::NONE && (defenders & piece_mask).is_not_empty() {
            defender = piece;
        }

        if attacker != Piece::NONE && defender != Piece::NONE {
            break;
        }
    }

    (attacker, defender)
}