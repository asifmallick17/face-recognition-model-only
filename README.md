1. at first create a venv and activate this.
2. Then install all the requirements given in the 'requirements.txt' file in the virtual environment using this command - 'pip install -r requirements.txt'.
3. create a 'datasets' folder in which  200 images of every person is present.
4.  Then run the 'generate_embeddings.py' file and it will create 'embeddings.npy' and 'labels.npy' 
5. Then run 'train_svm.py' and it will create 'face_recognition_model.pkl'
6. Then run 'predict_students.py' and the window will open where the faces are recognized. Tap any key to close the window.