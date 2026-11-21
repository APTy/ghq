from public.engine import AIRBORNE_INFANTRY, ARMORED_ARTILLERY, ARMORED_INFANTRY, ARTILLERY, BLUE, HEAVY_ARTILLERY, INFANTRY, RED, BaseBoard, ReserveFleet
import numpy as np

def serialize_board(board: BaseBoard) -> np.ndarray:
    """
    Return a float32 tensor of shape (C, 8, 8).
    """
    planes = []
    planes.extend(planes_binary(board))
    planes.extend(planes_scalar(board))

    # shape (C,8,8)
    return np.stack(planes, axis=0)

def planes_binary(board: BaseBoard):
    planes = [
        # units
        bb_to_plane(board.occupied),
        bb_to_plane(board.infantry),
        bb_to_plane(board.armored_infantry),
        bb_to_plane(board.airborne_infantry),
        bb_to_plane(board.artillery),
        bb_to_plane(board.armored_artillery),
        bb_to_plane(board.heavy_artillery),
        bb_to_plane(board.hq),
        # colors
        bb_to_plane(board.occupied_co[0]),
        bb_to_plane(board.occupied_co[1]),
        bb_to_plane(board.bombarded_co[0]),
        bb_to_plane(board.bombarded_co[1]),
        bb_to_plane(board.adjacent_infantry_squares_co[0]),
        bb_to_plane(board.adjacent_infantry_squares_co[1]),
        # orientation
        bb_to_plane(board.orientation_bit0),
        bb_to_plane(board.orientation_bit1),
        bb_to_plane(board.orientation_bit2),
        # mid-turn state
        bb_to_plane(board.turn_pieces),
        bb_to_plane(board.free_capture_clusters),
        bb_to_plane(board.free_capture_enemies),
        bb_to_plane(board.free_capture_num_allowed),
    ]
    return planes

MAX_TURN_MOVES = 3.0
MAX_TURN_AUTO_MOVES = 16.0

def planes_scalar(board: BaseBoard):
    # side-to-move:  all-ones if RED, zeros if BLUE
    stm = np.full((8, 8), 1.0 if board.turn == RED else 0.0, dtype=np.float32)

    # turn_moves is 0…3  → normalise to 0…1
    moves_scaled = np.full((8, 8), board.turn_moves / MAX_TURN_MOVES, dtype=np.float32)

    # turn_auto_moves is 0…16  → normalise to 0…16
    auto_moves_scaled = np.full((8, 8), board.turn_auto_moves / MAX_TURN_AUTO_MOVES, dtype=np.float32)

    planes = [stm, moves_scaled, auto_moves_scaled]
    planes.extend(fleet_planes_broadcast(board))
    return planes


MAX_RESERVE = 8

def fleet_planes_broadcast(board: BaseBoard) -> list[np.ndarray]:
    """12 planes: 6 piece types × (friendly, enemy)."""
    friendly, enemy = board.reserves   # already swapped by canonical_form

    planes = []
    for piece in (
        INFANTRY, ARMORED_INFANTRY, AIRBORNE_INFANTRY,
        ARTILLERY, ARMORED_ARTILLERY, HEAVY_ARTILLERY
    ):
        f_cnt = min(friendly.get_count(piece), MAX_RESERVE) / MAX_RESERVE
        e_cnt = min(enemy   .get_count(piece), MAX_RESERVE) / MAX_RESERVE
        planes.append(np.full((8, 8), f_cnt, dtype=np.float32))
        planes.append(np.full((8, 8), e_cnt, dtype=np.float32))
    return planes


def deserialize_board(tensor: np.ndarray) -> BaseBoard:
    """
    tensor: float32 (36, 8, 8) produced by serialize_board.
    Returns a fully populated BaseBoard.
    """
    if tensor.shape != (36, 8, 8):
        raise ValueError("Unexpected tensor shape; expected (36,8,8)")

    planes = tensor

    # -------- binary bitboards (same order used in planes_binary) -------
    bb = [plane_to_bb(p) for p in planes[:21]]

    new = BaseBoard(skip_init=True)
    new.clear_board()
    new.occupied                    = bb[0]
    new.infantry                    = bb[1]
    new.armored_infantry            = bb[2]
    new.airborne_infantry           = bb[3]
    new.artillery                   = bb[4]
    new.armored_artillery           = bb[5]
    new.heavy_artillery             = bb[6]
    new.hq                          = bb[7]
    new.occupied_co                 = [bb[8],  bb[9]]
    new.bombarded_co                = [bb[10], bb[11]]
    new.adjacent_infantry_squares_co= [bb[12], bb[13]]
    new.orientation_bit0            = bb[14]
    new.orientation_bit1            = bb[15]
    new.orientation_bit2            = bb[16]
    new.turn_pieces                 = bb[17]
    new.free_capture_clusters       = bb[18]
    new.free_capture_enemies        = bb[19]
    new.free_capture_num_allowed    = bb[20]

    # -------- scalar planes --------------------------------------------
    stm_plane          = planes[21]
    moves_scaled_plane = planes[22]
    auto_scaled_plane  = planes[23]

    new.turn           = RED if stm_plane[0, 0] > 0.5 else BLUE
    new.turn_moves     = int(round(moves_scaled_plane[0, 0] * MAX_TURN_MOVES))
    new.turn_auto_moves= int(round(auto_scaled_plane [0, 0] * MAX_TURN_AUTO_MOVES))

    # -------- reserve fleets -------------------------------------------
    friendly_counts = []
    enemy_counts    = []

    # 6 piece-types × (friendly, enemy)  → planes 24 … 35
    reserve_base = 24
    piece_order = (
        INFANTRY, ARMORED_INFANTRY, AIRBORNE_INFANTRY,
        ARTILLERY, ARMORED_ARTILLERY, HEAVY_ARTILLERY,
    )

    for i in range(6):
        f_plane = planes[reserve_base + 2 * i]
        e_plane = planes[reserve_base + 2 * i + 1]

        friendly_counts.append(int(round(f_plane[0, 0] * MAX_RESERVE)))
        enemy_counts   .append(int(round(e_plane[0, 0] * MAX_RESERVE)))

    friendly_fleet = ReserveFleet.from_ints(friendly_counts)
    enemy_fleet    = ReserveFleet.from_ints(enemy_counts)
    new.reserves   = [friendly_fleet, enemy_fleet]

    # ---- any final housekeeping your engine expects -------------------
    # e.g. recompute hash keys, update derived masks, etc.
    # new.refresh_caches()

    return new

def bb_to_plane_old(bb: int) -> np.ndarray:
    """
    Little-endian rank-file: bit0 = a1, … bit63 = h8
    """
    buf  = np.frombuffer(bb.to_bytes(8, 'little', signed=False), dtype=np.uint8)
    bits = np.unpackbits(buf, bitorder='little')          # <-- key change
    return bits.reshape(8, 8).astype(np.float32)

def plane_to_bb_old(plane: np.ndarray) -> int:
    """
    Inverse of bb_to_plane.
    """
    bits   = plane.astype(np.uint8).reshape(-1)           # 64 values 0/1
    packed = np.packbits(bits, bitorder='little')         # <-- key change
    return int.from_bytes(packed.tobytes(), 'little', signed=False)

# ---------------------------------------------------------------------
# 1.  Build a guaranteed-(256,8) lookup table in pure integer logic.
#     Works on any NumPy version, no unpackbits quirks.
# ---------------------------------------------------------------------
_LUT8 = (((np.arange(256, dtype=np.uint16)[:, None])  # (256,1)
          >> np.arange(8, dtype=np.uint16))           # (  1,8)
         & 1).astype(np.float32)                      # -> (256,8) float32


# ---------------------------------------------------------------------
# 2.  Fast little-endian bitboard  ⇄  (8,8) plane
# ---------------------------------------------------------------------
def bb_to_plane(bb: int) -> np.ndarray:
    """64-bit little-endian bitboard → (8,8) float32 plane."""
    bytes_view = np.frombuffer(np.uint64(bb).tobytes(), dtype=np.uint8)  # (8,)
    return _LUT8[bytes_view].reshape(8, 8)   # 64 floats, no copy


def plane_to_bb(plane: np.ndarray) -> int:
    """Inverse of bb_to_plane."""
    bits   = plane.reshape(-1).astype(np.uint8)             # 0/1 bytes
    packed = np.packbits(bits, bitorder='little')           # 8 bytes
    return int(np.frombuffer(packed, dtype=np.uint64)[0])
