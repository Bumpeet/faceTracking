import cv2 
import face_recognition
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import shutil
import os
from tqdm import tqdm
import streamlit as st
import tempfile


def face_rec(img_arr):
    dets = face_recognition.face_locations(img_arr)
    embeds = face_recognition.face_encodings(img_arr, dets)
    return dets, embeds

def extract_embeddings(path):
    cap = cv2.VideoCapture(path)

    list_embeds = []
    image_no = 0
    frame_no = 0
    
    shutil.rmtree("sub_images",ignore_errors=True)
    os.makedirs("sub_images", exist_ok=True)
    
    while cap.isOpened():
        _, frame = cap.read()

        if frame_no%20==0:
            try:
                faces, embeds = face_rec(frame)

                for i, val in enumerate(faces):
                    sub_img = frame[val[0]:val[2],val[3]:val[1],:]
                    cv2.imwrite(f'sub_images/{image_no}.jpg',sub_img)
                    list_embeds.append(embeds[i])
                    image_no += 1

            except Exception as e:
                break

        frame_no +=1
        # print(frame_no)

    cap.release()
    return list_embeds

def clustering(embeds):
    clusters = [4,5,6,7,8,9,10]
    scores_list = []
    clusters_list = []

    for i in clusters:
        model = KMeans(i)
        clusters = model.fit_predict(embeds)
        scores_list.append(silhouette_score(embeds,clusters))
        clusters_list.append(clusters)

    best_model_no = np.argmax(np.array(scores_list))
    best_model_clusters = clusters_list[best_model_no]
    n_clusters = len(set(best_model_clusters))

    print("Choosing the optimal cluster number and the number of clusters are: ", n_clusters)

    shutil.rmtree('master',ignore_errors=True)
    os.makedirs('master',exist_ok=True)


    for i in range(n_clusters):
        os.makedirs(f"master/{i}",exist_ok=True)

    for i, val in tqdm(enumerate(best_model_clusters),"moving the images into the clustered folders"):
        shutil.copy(f'sub_images/{i}.jpg',f'master/{val}')



def main():
    st.title("OpenCV VideoCapture with Streamlit")
    st.write("Upload a video and see its frames using OpenCV")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_filename = temp_file.name
            temp_file.write(uploaded_file.read())

            embeds = extract_embeddings(temp_filename)
            clustering(embeds)

            # Remove the temporary video file
            os.remove(temp_filename)



if __name__=="__main__":
    # pat = r"D:\Chetan\Downloads\video.mp4"
    # embeds = extract_embeddings(pat)
    # print("=====Extracted images and embeddings======")
    # clustering(embeds)
    main()




