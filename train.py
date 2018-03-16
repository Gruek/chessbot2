from db_trainer import DBTrainer
from bot import ChessBot

chbot = ChessBot()
trainer_standard = DBTrainer(chbot=chbot)
trainer_pro = DBTrainer(db_path='/data/kru03a/chbot/data/moves.db', chbot=chbot)

for i in range(30):
    print('TEST PRO #', i)
    trainer_pro.test()
    print('EPOCH PRO #', i)
    trainer_pro.epoch()

    print('TEST #', i)
    trainer_standard.test()
    print('EPOCH #', i)
    trainer_standard.epoch()
