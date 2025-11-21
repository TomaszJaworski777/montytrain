use bullet::game::{formats::bulletformat::ChessBoard, inputs};
use chess::{Bitboard, Piece, Rays};

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        BASE_INPUTS + STATE_INPUTS
    }

    fn max_active(&self) -> usize {
        64
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

        let (diag, ortho) = board.generate_pin_masks(board.side());
        let defender_pin_mask = diag | ortho;

        let (diag, ortho) = board.generate_pin_masks(board.side().flipped());
        let attack_pin_mask = diag | ortho;

        let horizontal_mirror = if board.king_square(board.side()).file() > 3 {
            7
        } else {
            0
        };

        let occ = board.occupancy();
        occ.map(|square| {
            let piece = board.piece_on_square(square);
            let color = board.color_on_square(square);

            let attack_pin_mask = attack_pin_mask & !Rays::get_ray(square, board.king_square(board.side().flipped()));

            let all_attackers = board.all_attackers_to_square(occ, square);
            let attackers = all_attackers & board.occupancy_for_side(board.side().flipped()) & !attack_pin_mask;
            let defenders = all_attackers & board.occupancy_for_side(board.side()) & !defender_pin_mask;

            let (attacker, defender) = attacker_defender(board, attackers, defenders);

            let piece_index = 64 * (u8::from(piece) - u8::from(Piece::PAWN)) as usize;
            let base = [384, 0][usize::from(color == board.side())] + piece_index + (usize::from(square) ^ horizontal_mirror);
            let mut feat = base;

            if attacker != Piece::NONE {
                feat += 768 * (usize::from(attacker) + 1)
            }

            if defender != Piece::NONE {
                feat += 768 * 7 * (usize::from(defender) + 1)
            }

            f(feat, feat);

            let feat = BASE_INPUTS + 768 * calculate_state(piece, attacker, defender, attackers, defenders) + base;
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

const BASE_INPUTS: usize = 768 * 7 * 7;
const STATE_INPUTS: usize = 768 * 6;

fn attacker_defender(board: &chess::ChessBoard, attackers: Bitboard, defenders: Bitboard) -> (Piece, Piece) {
    let attacker = lowest_value_piece(board, attackers);
    let defender = lowest_value_piece(board, defenders);
    (attacker, defender)
}

fn lowest_value_piece(board: &chess::ChessBoard, mask: Bitboard) -> Piece {
    if mask.is_empty() {
        return Piece::NONE;
    }

    if (mask & board.piece_mask(Piece::PAWN)).is_not_empty() {
        Piece::PAWN
    } else if (mask & board.piece_mask(Piece::KNIGHT)).is_not_empty() {
        Piece::KNIGHT
    } else if (mask & board.piece_mask(Piece::BISHOP)).is_not_empty() {
        Piece::BISHOP
    } else if (mask & board.piece_mask(Piece::ROOK)).is_not_empty() {
        Piece::ROOK
    } else if (mask & board.piece_mask(Piece::QUEEN)).is_not_empty() {
        Piece::QUEEN
    } else if (mask & board.piece_mask(Piece::KING)).is_not_empty() {
        Piece::KING
    } else {
        Piece::NONE
    }
}

fn calculate_state(victim: Piece, lowest_attacker: Piece, lowest_defender: Piece, attackers: Bitboard, defenders: Bitboard) -> usize {
    let atk_cnt = attackers.pop_count();
    let def_cnt = defenders.pop_count();

    if atk_cnt + def_cnt == 0 {
        return 5;
    }

    if def_cnt == 0 && atk_cnt > 0 {
        return 0; 
    }
    
    if atk_cnt == 0 && def_cnt > 0 {
        return 4;
    }

    let v_victim = u8::from(victim) as usize;
    let v_attacker = u8::from(lowest_attacker) as usize;
    let v_defender = u8::from(lowest_defender) as usize;
    
    if atk_cnt == 1 && def_cnt == 1 {
        if v_attacker < v_victim {
            return 1;
        } else if v_attacker > v_victim {
            return 3;
        } else {
            return 2;
        }
    }

    if atk_cnt > 1 && def_cnt <= atk_cnt && v_victim + v_defender > v_attacker {
        return 1;
    }

    if atk_cnt > 1 && def_cnt == atk_cnt && v_victim + v_defender < v_attacker {
        return 3;
    }
    
    let diff = (def_cnt as i32) - (atk_cnt as i32);

    if diff < 0 {
        return 1;
    } else if diff == 0 {
        if v_attacker < v_victim {
             return 1;
        }
        return 2;
    } else {
        return 3
    }
}