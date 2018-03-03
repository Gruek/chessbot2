from db_trainer import DBTrainer

trainer = DBTrainer()
for i in range(30):
    print('EPOCH #', i)
    trainer.epoch()
    print('TEST #', i)
    trainer.test()
