from flask import render_template, request, flash, redirect, Blueprint, url_for
from werkzeug import secure_filename
import os
import re
from flask import current_app
from spamfilter.models import db, File
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from spamfilter.forms import InputForm
from spamclassifiers import spamclassifier as spc
import glob
import numpy as np


spam_api = Blueprint('SpamAPI', __name__)


def allowed_file(filename, extensions=None):
    '''
    'extensions' is either None or a list of file extensions.

    If a list is passed as 'extensions' argument, check if 'filename' contains
    one of the extension provided in the list and return True or False respectively.

    If no list is passed to 'extensions' argument, then check if 'filename' contains
    one of the extension provided in list 'ALLOWED_EXTENSIONS', defined in 'config.py',
    and return True or False respectively.
    '''
    extn = filename.rsplit(".", 1)[1]
    if extensions:
        if extn in extensions:
            return True
        else:
            return False
    else:
        if extn in current_app.config['ALLOWED_EXTENSIONS']:
            return True
        else:
            return False


@spam_api.route('/')
def index():
    '''
    Renders 'index.html'
    '''
    return render_template('index.html')


@spam_api.route('/listfiles/<success_file>/')
@spam_api.route('/listfiles/')
def display_files(success_file=None):
    '''
    Obtain the filenames of all CSV files present in 'inputdata' folder and
    pass it to template variable 'files'.

    Renders 'filelist.html' template with values  of variables 'files' and 'fname'.
    'fname' is set to value of 'success_file' argument.

    if 'success_file' value is passed, corresponding file is highlighted.
    '''
    path = current_app.config['INPUT_DATA_UPLOAD_FOLDER']
    files =[filename for filename in os.listdir(path) if (".csv" in filename)]
    fname = success_file

    '''extension = 'csv'
    os.chdir(path)
    files = glob.glob('*.{}'.format(extension))
    fname=''
    if success_file:
      fname = success_file'''

    #print('inside display_files')

    return render_template('fileslist.html', fname=fname, files=files)


def validate_input_dataset(input_dataset_path):
    '''
    Validate the following details of an Uploaded CSV file

    1. The CSV file must contain only 2 columns. If not display the below error message.
    'Only 2 columns allowed: Your input csv file has '+<No_of_Columns_found>+ ' number of columns.'

    2. The column names must be "text" nad "spam" only. If not display the below error message.
    'Differnt Column Names: Only column names "text" and "spam" are allowed.'

    3. The 'spam' column must conatin only integers. If not display the below error message.
    'Values of spam column are not of integer type.'

    4. The values of 'spam' must be either 0 or 1. If not display the below error message.
    'Only 1 and 0 values are allowed in spam column: Unwanted values ' + <Unwanted values joined by comma> + ' appear in spam column'

    5. The 'text' column must contain string values. If not display the below error message.
    'Values of text column are not of string type.'

    6. Every input email must start with 'Subject:' pattern. If not display the below error message.
    'Some of the input emails does not start with keyword "Subject:".'

    Return False if any of the above 6 validations fail.

    Return True if all 6 validations pass.
    '''
    valid = False
    first_line = ""
    #TODO add handling if not able to open file
    print('start validate method')
    with open(input_dataset_path) as infile:
        first_line = infile.readline()

    columns = first_line.split(",")
    print('file opened in validate method')

    No_of_Columns_found = len(columns)
    if No_of_Columns_found != 2:
        flash('Only 2 columns allowed: Your input csv file has ' +
              str(No_of_Columns_found) + ' number of columns.')
    elif not ((columns[0] == 'text') and (columns[1].strip() == 'spam')):
        flash('Differnt Column Names: Only column names "text" and "spam" are allowed.')
    else:
        df = pd.read_csv(input_dataset_path, dtype={
                         "text": "str", "spam": "int"})
        column0 = df.text.values
        column1 = df.spam.values
        #print(type(column1[0]))
        # df.info()
        if not all([isinstance(x, np.int64) or isinstance(x, np.int32) for x in column1]):
            flash('Values of spam column are not of integer type.')
        elif not all([((x == 0) or (x == 1)) for x in column1]):
            temp = [x for x in column1 if x != 0 or x != 1]
            Unwantedvalues = ','.join(map(str, temp))
            flash('Only 1 and 0 values are allowed in spam column: Unwanted values ' +
                  Unwantedvalues + ' appear in spam column')
        elif not all(isinstance(x, str) for x in column0):
            flash('Values of text column are not of string type.')
        elif not all([x.startswith('Subject:') for x in column0]):
            flash('Some of the input emails does not start with keyword "Subject:".')
        else:
            valid = True
    print(valid)
    return valid


@spam_api.route('/upload/', methods=['GET', 'POST'])
def file_upload():
    '''
    If request is GET, Render 'upload.html'

    If request is POST, capture the uploaded file a

    check if the uploaded file is 'csv' extension, using 'allowed_file' defined above.

    if 'allowed_file' returns False, display the below error message and redirect to 'upload.html' with GET request.
    'Only CSV Files are allowed as Input.'

    if 'allowed_file' returns True, save the file in 'inputdata' folder and
    validate the uploaded csv file using 'validate_input_dataset' defined above.

    if 'validate_input_dataset' returns 'False', remove the file from 'inputdata' folder,
    redirect to 'upload.html' with GET request and respective error message.

    if 'validate_input_dataset' returns 'True', create a 'File' object and save it in database, and
    render 'display_files' template with template varaible 'success_file', set to filename of uploaded file.

    '''
    if request.method == 'POST':
        if ('file' not in request.files):
            flash('No file part')
            return render_template('upload.html')
        else:
            fileo = request.files['file']
            if not allowed_file(fileo.filename, ['csv']):
                flash('Only CSV Files are allowed as Input.')
                return render_template('upload.html')
            else:

              filename = secure_filename(fileo.filename)
              path = os.path.join(
                  current_app.config['INPUT_DATA_UPLOAD_FOLDER'], filename)
              print('file part32'+filename+path)
              fileo.save(path)
              #print('file saved')
              validated = validate_input_dataset(path)
              if not validated:
                  os.remove(path)
                  
                  return render_template('upload.html')
              else:
                  #print('file part3'+filename+current_app.config['INPUT_DATA_UPLOAD_FOLDER'])
                  fileobj = File(name = filename,filepath = current_app.config['INPUT_DATA_UPLOAD_FOLDER'])
                  '''fileobj.name = filename
                  fileobj.filepath = current_app.config['INPUT_DATA_UPLOAD_FOLDER']'''
                  db.session.add(fileobj)
                  db.session.commit()
                  print('db commit done')
                  print(File.query.all())

                  return display_files(success_file=filename)

    else:
        return render_template('upload.html')


def validate_input_text(intext):
    '''
    Validate the following details of input email text, provided for prediction.

    1. If the input email text contains more than one mail, they must be separated by atleast one blank line.

    2. Every input email must start with 'Subject:' pattern.

    Return False if any of the two validations fail.

    If all validations pass, Return an Ordered Dicitionary, whose keys are first 30 characters of each
    input email and values being the complete email text.
    '''
    count = 0
    if isinstance(intext, str):
        count = intext.count('Subject:')
    else:
        count = len(intext)

    if (count == 1):

        if isinstance(intext, list):
            if intext[0].count('Subject:') == 1:
                text = intext[0]
            else:
                return False
        else:
            text = intext
        if text.startswith("Subject:"):
            od = OrderedDict()
            od[text[0:30]] = text
            return od
        else:
            return False
    elif count > 1:
        # TODO
        liststr = []
        if isinstance(intext, str):
            liststr = intext.split('\\n')
        else:
            liststr = intext
        od = OrderedDict()
        for txt in liststr:
            if txt.count('Subject:') > 1:
                return False
            elif len(txt.strip()) > 0:
                od[txt[0:30]] = txt
        return od
    else:
        return False


@spam_api.route('/models/<success_model>/')
@spam_api.route('/models/')
def display_models(success_model=None):
    '''
    Obtain the filenames of all machine learning models present in 'mlmodels' folder and
    pass it to template variable 'files'.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Consider only the model and not the word_features.pk files.

    Renders 'modelslist.html' template with values  of varaibles 'files' and 'model_name'.
    'model_name' is set to value of 'success_model' argument.

    if 'success_model value is passed, corresponding model file name is highlighted.
    '''
    path = current_app.config['ML_MODEL_UPLOAD_FOLDER']
    files =[filename for filename in os.listdir(path) if not ("word_features.pk" in filename)]
    '''print(files)'''
    '''os.chdir(path)
    fs = glob.glob('*.pk')
    files = [f for f in fs if not ("word_features" in f)]'''
    model_name = success_model

    return render_template('modelslist.html', model_name=model_name, files=files)


def isFloat(value):
    '''
    Return True if <value> is a float, else return False
    '''
    try:
        float(value)
        return True
    except ValueError:
        return False


def isInt(value):
    '''
    Return True if <value> is an integer, else return False
    '''
    try:
        int(value)
        return True
    except ValueError:
        return False


@spam_api.route('/train/', methods=['GET', 'POST'])
def train_dataset():
    '''
    If request is of GET method, render 'train.html' template with template variable 'train_files',
    set to list if csv files present in 'inputdata' folder.

    If request is of POST method, capture values associated with
    'train_file', 'train_size', 'random_state', and 'shuffle'

    if no 'train_file' is selected, render the same page with GET Request and below error message.
    'No CSV file is selected'

    if 'train_size' is None, render the same page with GET Request and below error message.
    'No value provided for size of training data set.'

    if 'train_size' value is not float, render the same page with GET Request and below error message.
    'Training Data Set Size must be a float.

    if 'train_size' value is not in between 0.0 and 1.0, render the same page with GET Request and below error message.
    'Training Data Set Size Value must be in between 0.0 and 1.0'

    if 'random_state' is None,render the same page with GET Request and below error message.
    'No value provided for random state.''

    if 'random_state' value is not an integer, render the same page with GET Request and below error message.
    'Random State must be an integer.'

    if 'shuffle' is None, render the same page with GET Request and below error message.
    'No option for shuffle is selected.'

    if 'shuffle' is set to 'No' when 'Startify' is set to 'Yes', render the same page with GET Request and below error message.
    'When Shuffle is No, Startify cannot be Yes.'

    If all input values are valid, build the model using submitted paramters and methods defined in
    'spamclassifier.py' and save the model and model word features file in 'mlmodels' folder.

    NOTE: These models are generated from uploaded CSV files, present in 'inputdata'.
    So if ur csv file names is 'sample.csv', then when you generate model
    two files 'sample.pk' and 'sample_word_features.pk' will be generated.

    Finally render, 'display_models' template with value of template varaible 'success_model'
    set to name of model generated, ie. 'sample.pk'
    '''

    if request.method == 'POST':
        train_file = request.form.get('train_file')
        train_size = request.form.get('train_size')
        random_state = request.form.get('random_state')
        shuffle = request.form.get('shuffle')
        stratify = request.form.get('stratify')

        if not train_file:
          flash('No CSV file is selected')
          return redirect(request.url)
        elif train_size is None:
          flash('No value provided for size of training data set.')
          return redirect(request.url)
        elif not isFloat(train_size):
          flash('Training Data Set Size must be a float.')
          return redirect(request.url)
        elif not (float(train_size) >=0.0 and float(train_size)<= 1.0):
          flash('Training Data Set Size Value must be in between 0.0 and 1.0.')
          return redirect(request.url)
        elif random_state is None:
          flash('No value provided for random state.')
          return redirect(request.url)
        elif not isInt(random_state):
          flash('Random State must be an integer.')

          return redirect(request.url)
        elif shuffle is None:
          flash('No option for shuffle is selected.')
          return redirect(request.url)
        elif shuffle=='N' and stratify=='Y':
          flash('When Shuffle is No, Startify cannot be Yes.')
          return redirect(request.url)

        path = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], train_file)
        df=pd.read_csv(path, header=0).values
        classifier, word_features = spc.SpamClassifier().train(df[:,0:1],df[:,1])
        prefixname = os.path.splitext(os.path.basename(train_file))[0]
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], prefixname+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],prefixname +'_word_features.pk')

        '''pickle.dump(classifier, open(model_file, 'wb'))
        pickle.dump(word_features, open(model_word_features_file, 'wb'))'''

        filehandler = open(model_file,"wb")
        pickle.dump(classifier,filehandler)
        filehandler.close()

        filehandler = open(model_word_features_file,"wb")
        pickle.dump(word_features,filehandler)
        filehandler.close()


        return display_models(success_model=prefixname+".pk")

    else:
      path = current_app.config['INPUT_DATA_UPLOAD_FOLDER']

      train_files =[filename for filename in os.listdir(path) if (".csv" in filename)]
      return render_template('train.html', train_files=train_files)


@spam_api.route('/results/')
def display_results():
    '''
    Read the contents of 'predictions.json' and pass those values to 'predictions' template varaible

    Render 'displayresults.html' with value of 'predictions' template variable.
    '''
    path = os.path.join(
        current_app.config['INPUT_DATA_UPLOAD_FOLDER'], 'predictions.json')
    with open(path) as infile:
        lines = json.load(infile)

    list = [(k, v) for k, v in lines.items()]
    '''print('prediction result')
    print(list)'''
    return render_template('displayresults.html', predictions=list)


@spam_api.route('/predict/', methods=['GET', "POST"])
def predict():
    '''
    If request is of GET method, render 'emailsubmit.html' template with value of template
    variable 'form' set to instance of 'InputForm'(defined in 'forms.py').
    Set the 'inputmodel' choices to names of models (in 'mlmodels' folder), with out extension i.e .pk

    If request is of POST method, perform the below checks

    1. If input emails is not provided either in text area or as a '.txt' file, render the same page with GET Request and below error message.
    'No Input: Provide a Single or Multiple Emails as Input.'

    2. If input is provided both in text area and as a file, render the same page with GET Request and below error message.
    'Two Inputs Provided: Provide Only One Input.'

    3. In case if input is provided as a '.txt' file, save the uploaded file into 'inputdata' folder and read the
     contents of file into a variable 'input_txt'

    4. If input provided in text area, capture the contents in the same variable 'input_txt'.

    5. validate 'input_txt', using 'validate_input_text' function defined above.

    6. If 'validate_input_text' returns False, render the same page with GET Request and below error message.
    'Unexpected Format : Input Text is not in Specified Format.'


    7. If 'validate_input_text' returns a Ordered dictionary, choose a model and perform prediction of each input email using 'predict' method defined in 'spamclassifier.py'

    8. If no input model is choosen, render the same page with GET Request and below error message.
    'Please Choose a single Model'

    9. Convert the ordered dictionary of predictions, with 0 and 1 values, to another ordered dictionary with values 'NOT SPAM' and 'SPAM' respectively.

    10. Save thus obtained predictions ordered dictionary into 'predictions.json' file.

    11. Render the template 'display_results'

    '''


    if request.method == 'POST':
        inputemail = request.form.get('inputemail')
        inputfile = request.files['inputfile']
        inputmodel = request.form.get('inputmodel')


        if not (inputemail) and not (inputfile) :
          flash('No Input: Provide a Single or Multiple Emails as Input.')
          return redirect(request.url)
        elif (inputemail) and (inputfile):
          flash('Two Inputs Provided: Provide Only One Input.')
          return redirect(request.url)
        elif inputfile and inputfile.filename.endswith('.txt'):

          filename = secure_filename(inputfile.filename)
          path = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], filename)
          inputfile.save(path)
          with open(path) as infile:
            input_txt = infile.readlines()
        elif inputemail:
          input_txt = inputemail

        valid = validate_input_text(input_txt)
        '''print('valid')
        print(valid)'''
        if valid == False:
          flash('Unexpected Format : Input Text is not in Specified Format.')
          return redirect(request.url)
        else:
          if not inputmodel:
            flash('Please Choose a single Model')
            return redirect(request.url)
          else:
            loaded_model = spc.SpamClassifier()
            loaded_model.load_model(inputmodel)
            result = loaded_model.predict(valid)
            '''print('result')
            print(result)'''
            for k, v in result.items():
                if v==0:
                  result[k] = 'NOT SPAM'
                else:
                  result[k] = 'SPAM'
            path = os.path.join(current_app.config['INPUT_DATA_UPLOAD_FOLDER'], 'predictions.json')
            with open(path, 'w') as f:
              json.dump(result, f)


        return display_results()


    elif request.method == 'GET':
      form = InputForm()
      #print('predict entry get')
      path = current_app.config['ML_MODEL_UPLOAD_FOLDER']
      list_of_files =[(filename.rsplit(".", 1)[0],filename.rsplit(".", 1)[0] ) for filename in os.listdir(path) if not ("word_features" in filename)]
      #print(list_of_files)
      form.inputmodel.choices = list_of_files
      form.process()

      '''path = current_app.config['ML_MODEL_UPLOAD_FOLDER']
      os.chdir(path)
      files = glob.glob('*.pk')

      choices = [(os.path.splitext(os.path.basename(f))[0],os.path.splitext(os.path.basename(f))[0]) for f in files if not ("word_features" in f)]

      form.inputmodel.choices = choices
      form.process()'''

      return render_template('emailsubmit.html', form=form )
