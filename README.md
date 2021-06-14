# NLP-EMAIL-SPAM-FLASK

Repo has flask based web app created for a usecase of classifying email as spam or not.
<h1 class="title"> Spam Filter </h1> <br>
<p class="subtitle"> Welcome to SPAM FILTER application. This application can be used to perform the following tasks.
    <li> Upload a data set of labelled emails. </li>
    <li> Train the uploaded emails dataset based on our built in Spam Classifier. </li>
    <li> Build a Machine Learning model, capable of differentiating spam emails and non-spam emails, based on our built-in Spam Classifier. </li>
    <li> Use the build model to predict if input emails are SPAM or NOT SPAM. </li>
</p>
<p>
For complete usage of this application, view the rules defined below for various tasks.
</p>

<h3> Rules for Uploading a Data Set </h3>
<li> Data Set must be a csv file. </li>
<li> Data Set must contain two columns, with column names 'text', and 'spam' respectively. </li>
<li> Values in 'text' column must be email text starting with keyword 'Subject:' . </li>
<li> Values in 'spam' column must be either 0 or 1. 1 must indicate a SPAM email and 0 must indicate NON SPAM email.</li>
<br>


<h3> Rules for Training and Building a Model </h3>
<li> Before Training Please upload a data set. </li>
<li> Choose one of the uploaded CSV files for training.</li>
<li> Provide inputs for parameters  Test Data Set Size, Random State, and Shuffle. </li>
    <br>
    <h3> Rules for Predicting </h3>
    <li> A single or multiple emails can be provided as input.</li>
    <li> Each input email must start with keyword 'Subject:' .</li>
    <li> If multiple emails are provided, then each mail must be separated from another by atleast a single blank line.</li>
    <li> The input can be provided either by pasting text in available area or can be submitted in a text file.</li>
</div>
