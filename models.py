from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import Column, Integer, String
db = SQLAlchemy()
migrate = Migrate()

class File(db.Model):
    '''
    Define the following three columns
    1. id - Which stores a file id
    2. name - Which stores the file name
    3. filepath - Which stores path of file,
    '''

    file_id = db.Column(db.Integer,primary_key=True, autoincrement=True)
    name = db.Column('name',db.String(500))
    filepath = db.Column('filepath', db.Unicode)


    def __rep__(self):
        return "<File : {}>".format(self.name)

