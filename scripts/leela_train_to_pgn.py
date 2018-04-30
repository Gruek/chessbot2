import tarfile
import os
import numpy as np
import chess
from chess import pgn
from multiprocessing import Pool

outputdir = './leela_pgn'
inputdir = './leela_data'
ensure_valid = False # Check for move legality (requires generating moves - slow)
include_header = True # include header in PGN output?
single_file = True # output to one file
games_in_file = 10000

try:
    os.makedirs(outputdir)
except OSError as e:
    if not os.path.exists(outputdir):
        raise

indexed_pieces = list(enumerate(['P', 'N', 'B', 'R', 'Q', 'K']))
columns = 'abcdefgh'


def getbps_result(data):
    '''Get my-move bit-planes:
        returns an array (number of move elements) of arrays (each with 2 elements, 1 per side)
            of arrays (1 per piece, 6 total) of 8-bytes'''
    chunksize = 8276
    offset = 4+7432
    numplanes = 6
    bps = []
    assert len(data)%chunksize == 0
    for i in range(len(data)//chunksize):
        sidebps = []
        for side in range(2):
            planes = data[i*chunksize+offset + side*numplanes*8
                          :i*chunksize + offset + numplanes*8 + side*numplanes*8]
            piecebbs = []
            for plane in range(numplanes):
                piecebbs.append(planes[plane*8:(plane+1)*8])
            sidebps.append(piecebbs)
        bps.append(sidebps)
    result_offset = chunksize - 1
    result = np.int8(data[(i-1)*chunksize + result_offset])
    if i%2==0:
        # I am white -- result seems to be flipped from current player
        white_result = -result
    else:
        white_result = result
    # print(i%2, )
    # print(i%2, np.int8(data[i*chunksize + result_offset]))
    return bps, white_result

def bp_to_array(bp, flip):
    '''Given an 8-byte bit-plane, convert to uint8 numpy array'''
    if not flip:
        return np.unpackbits(bytearray(bp)).reshape(8, 8)
    else:
        return np.unpackbits(bytearray(bp)).reshape(8, 8)[::-1]


def convert_to_move(planes1, planes2, move_index):
    '''Given two arrays of 8-byte bit-planes, convert to a move
    Also need the index of the first move (0-indexed) to determine how to flip the board.
    '''
    
    current_player = move_index % 2
    
    # Check for K moves first bc of castling, pawns last for promotions...
    for idx, piece in reversed(indexed_pieces):
        arr1 = bp_to_array(planes1[idx], current_player==1)
        arr2 = bp_to_array(planes2[idx], current_player==0)
        if not np.array_equal(arr1, arr2):
            rowfrom, colfrom = np.where(arr1 & ~arr2)
            rowto, colto = np.where(~arr1 & arr2)
            promotion = ''
            if not len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1:
                # This must be a pawn promotion...
                assert (len(colfrom)==len(rowfrom)==0)
                # Find where the pawn came from
                p_arr1 = bp_to_array(planes1[0], current_player==1)
                p_arr2 = bp_to_array(planes2[0], current_player==0)
                rowfrom, colfrom = np.where(p_arr1 & ~p_arr2)
                promotion = piece.lower()
            assert len(colfrom)==len(rowfrom)==len(colto)==len(rowto)==1
            rowfrom, colfrom = rowfrom[0], colfrom[0]
            rowto, colto = rowto[0], colto[0]
            uci = '{}{}{}{}{}'.format(columns[colfrom], rowfrom+1,
                                    columns[colto], rowto+1, promotion)
            return piece, uci
    else:
        raise Exception("I shouldn't be here")

def getpgn(data, name):
    global game, node, white_result
    game = chess.pgn.Game()
    game.headers["Event"] = name
    node = game
    bps, white_result = getbps_result(data)
    if white_result==1:
        game.headers["Result"] = "1-0"
    elif white_result==-1:
        game.headers["Result"] = "0-1"
    elif white_result==0:
        game.headers["Result"] = "1/2-1/2"
    else:
        print(white_result)
        raise Exception("Bad result")
    for move in range(len(bps)-1):
        piece, uci = convert_to_move(bps[move][0], bps[move+1][1], move)
        move = chess.Move.from_uci(uci)
        if ensure_valid:
            valid_moves = list(node.board().generate_legal_moves())
            assert move in valid_moves
        node = node.add_variation(move)
    return str(game)


def write_pgn(pgn, name, pgnfile):
    if pgnfile:
        print(pgn, file=pgnfile)
        print('\n', file=pgnfile)
    else:
        pgnfilename = name + '.pgn'
        pgnfilename = os.path.join(outputdir, pgnfilename)
        with open(pgnfilename, 'w') as pgnfile:
            print(pgn, file=pgnfile)

def process_file(filename):
    try:
        if single_file:
            pgnfilename = os.path.basename(filename).split('.', 1)[0] + '.pgn'
            pgnfilename = os.path.join(outputdir, pgnfilename)
            if os.path.exists(pgnfilename):
                print("File already exists - skipping", filename)
                return
        with open(pgnfilename, 'w') as pgnfile:
            with tarfile.open(filename) as f:
                for idx, member in enumerate(f):
                    if idx%100==0:
                        # print('\n{}/{}'.format(idx, games_in_file), end='')
                        if single_file:
                            pgnfile.flush()
                    # print('.', end='')
                    data = f.extractfile(member).read()
                    pgn = getpgn(data, member.name)
                    if not include_header:
                        # This chops off the header of the pgn string
                        pgn = pgn.rsplit('\n', 1)[-1]
                    write_pgn(pgn, member.name, pgnfile)
    except AssertionError as e:
        print(e)
        print("Error skipping", filename)
         
if __name__ == '__main__':
    with Pool(processes=10) as pool:    
        for filename in os.listdir(inputdir):
            filename = os.path.join(inputdir, filename)
            print(filename)
            pool.apply_async(process_file, [filename])
        pool.close()
        pool.join()
