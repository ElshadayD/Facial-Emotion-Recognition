
from cgitb import reset
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from flask import Flask, request, render_template,url_for, session,redirect,flash
import statistics as st
import tensorflow as tf
from time import sleep
from keras.preprocessing import image
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import time
import numpy as np
import random
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'registration'
app.config['UPLOAD_FOLDER']='/static/dataset'

# Intialize MySQL
mysql = MySQL(app)


@app.route('/transition',methods=['POST','GET'])
def transition():
        return render_template("login.html")
@app.route('/login', methods=['GET', 'POST'])
def login():
# Output message if something goes wrong...
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            flash("Incorrect username/password!", "danger")
    return render_template('login.html')



# http://localhost:5000/pythinlogin/register 
# This will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # cursor.execute('SELECT * FROM accounts WHERE username = %s', (username))
        cursor.execute( "SELECT * FROM users WHERE username LIKE %s", [username] )
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            flash("Account already exists!", "danger")
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash("Invalid email address!", "danger")
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash("Username must contain only characters and numbers!", "danger")
        elif not username or not password or not email:
            flash("Incorrect username/password!", "danger")
        else:
        # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (username,email, password))
            mysql.connection.commit()
            flash("You have successfully registered!", "success")
        

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash("Please fill out the form!", "danger")
    # Show registration form with message (if any)
    return render_template('login.html')

# http://localhost:5000/pythinlogin/home 
# This will be the home page, only accessible for loggedin users
@app.route('/')
def mainpage():
    return render_template('mainpage.html')
   
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'],title="Home")
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))    


@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    classifier = load_model('./Emotion_Detection.h5')
    classifier.make_predict_function()
    output = []

    class_labels = ['angry', 'happy', 'neutral', 'sad', 'surprised']
    i=0
    cap = cv2.VideoCapture(0)
    while i<=30:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]

                label = class_labels[preds.argmax()]
                label_position = (x, y)
                output.append(label)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        i=i+1

        cv2.imshow('Emotion Detector', frame)
        cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html", final_output=final_output1)

IMAGE_FOLDER=os.path.join('static','images')
@app.route("/train", methods=['GET', 'POST'])

def train():
    images=['h6.jpg','h1.jpg','h3.jpg','s1.jpg',
    's2.jpg','s2.jpg','n1.jpg','n3.jpg','a1.jpg',
    'h4.jpg','a3.jpg','su1.jpg','su2.jpg',
    'su3.jpg']
    sampled_list = random.sample(images, 14)
    img_path=os.path.join(IMAGE_FOLDER, random.choice(sampled_list))
    result=DeepFace.analyze(img_path,actions=['emotion'],enforce_detection=False)
    dominant=result[ 'dominant_emotion']

    
    return render_template("train.html", image=img_path,result=result,dominant=dominant)
   

@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")


@app.route('/movies/surprised', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("movieSurprised.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprised', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprised.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsNeutral.html")
@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")


if __name__ == "__main__":
    app.run(debug=True)



























