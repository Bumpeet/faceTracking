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
import time

def face_rec(img_arr):
    dets = face_recognition.face_locations(img_arr)
    embeds = face_recognition.face_encodings(img_arr, dets)
    return dets, embeds

def extract_embeddings(path,frame_skip):
    cap = cv2.VideoCapture(path)

    list_embeds = []
    list_dets = []
    frames = []
    image_no = 0
    frame_no = 0

    local_folder = "images"
    face_crops_folder = f'{local_folder}/sub_images'
    os.makedirs(face_crops_folder, exist_ok=True)

    # length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # time = length/frame_rate


    with st.spinner(f"Extracting embeddings from frames"):
        with st.empty():
    
            while cap.isOpened():
                ret, frame = cap.read()

                if ret==True and frame_no%frame_skip==0:
                    st.image(frame,f"Extracting faces from the frame-{frame_no} in the video", channels="BGR",width=480)
                    frames.append(frame)
                    try:
                        dets, embeds = face_rec(frame)

                        list_embeds.append(embeds)
                        list_dets.append(dets)

                        for i, val in enumerate(dets):
                            sub_img = frame[val[0]:val[2],val[3]:val[1],:]
                            cv2.imwrite(f'{face_crops_folder}/{image_no}.jpg',sub_img)
                            print(f'saved image - {image_no} to the \'{face_crops_folder}\' folder')
                            image_no += 1

                    except Exception as e:
                        st.exception(f"{e}",icon="⚠️")

                elif ret==False:
                    break

                frame_no +=1
                
            
            cap.release()
            st.empty()
    st.toast("Extracted Embeddings from all the frames of the video",icon="👨")

    return list_embeds, list_dets, frames

def clustering(embeds):
    '''
    This method helps in clustering the embeddings using the KMeans algorithm. The optimal number 
    of clusters will be chosen based on the Shilloute score.

    params:
    - embeds: list of embeddings of all the faces
    returns:
    - the best Kmeans model
    '''

    best_score = 0.0
    best_model = None

    list_embeds = []

    for embed in embeds:
        for emb in embed:
            list_embeds.append(emb)

    n_samples = len(list_embeds)

    with st.empty():
        progress_text = "Clustering the extracted embedding using KMeans."
        my_bar = st.progress(0, text=progress_text)

        for i in tqdm(range(2,n_samples,1),"Fitting the model with give set of clusters"):
            model = KMeans(i)
            clusters = model.fit_predict(list_embeds)
            score = silhouette_score(list_embeds,clusters)
            my_bar.progress(i + 1, text=progress_text)
            # print(score)
            if score > best_score:
                best_model = model
                best_score = score
        st.empty()

    st.toast("Finished clustering the embeddings",icon="✅")
    if best_model is None:
        st.warning("please upload a video contanining the human faces")
        st.stop()
    best_model_clusters = best_model.labels_
    n_clusters = np.max(best_model_clusters) + 1

    st.info(f"Found {n_clusters} unique faces among the video",icon="✅")

    print("The optimal number of clusters based on the shilloute score are: ", n_clusters)

    for i in range(n_clusters):
        os.makedirs(f"images/{i}",exist_ok=True)

    for i, val in tqdm(enumerate(best_model_clusters),"moving the images into the clustered folders"):
        shutil.copy(f'images/sub_images/{i}.jpg',f'images/{val}')

    return best_model

def create_temp_dirs():
    shutil.rmtree("images", ignore_errors=True)
    os.makedirs("images", exist_ok=True)
    # os.remove("output_video.mp4",)


def generate_video(embeds, dets, frames, model):
    '''
    Generates the video with bounding box and id's

    params:
    - embeds: list of embeddings of all the detections
    - dets: list of bbox of all the detections
    - model: K-Means model for predicting the cluster id
    '''



    width = frames[0].shape[1]
    height = frames[0].shape[0]

    out = cv2.VideoWriter('output_video.webm',cv2.VideoWriter_fourcc(*'VP90'), 5, (int(width), int(height)))

    with st.spinner("Creating the video file to display it"):

        for i, frame in enumerate(frames):
            for sub_embed, sub_det in zip(embeds[i], dets[i]):
                cv2.rectangle(frame,(sub_det[3], sub_det[0]),(sub_det[1], sub_det[2]),color=(0,0,255),thickness=2)
                cluster_id = model.predict(sub_embed.reshape(1,-1))
                cluster_id_str = str(cluster_id[0])
                # print(cluster_id_str, type(cluster_id_str))
                cv2.putText(frame,cluster_id_str,
                            (sub_det[3], sub_det[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            color=(0, 255, 0),
                            fontScale = 1,
                            thickness=2 )
                out.write(frame)


        out.release()

def main():

    uploaded_file = st.file_uploader("Choose a video file to run the face tracking, \
                                     make sure the video is less than 20 seconds for the faster results", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:

        create_temp_dirs()

        print("created the Temperory directories")
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(uploaded_file.read())

        place_holder = st.empty()

        skip_frames = st.slider('Use this slider to skip the frames for the faster performance', 0, 50)

        if skip_frames:

            print("Sending images to extract the embeddings")
            embeds, dets, frames = extract_embeddings(temp_filename, skip_frames )
            model = clustering(embeds)


            generate_video(embeds, dets, frames, model)

            with st.spinner("Reading the video file to display it"):


                video_file = open('output_video.webm', 'rb')
                video_bytes = video_file.read()
            st.balloons()

            st.video(video_bytes,format="video/webm")

            st.divider()
            st.write("Use this download button to download the clustered images")
            shutil.make_archive("images","zip","images")

            with open("images.zip", "rb") as fp:
                btn = st.download_button(
                    label="Download ZIP",
                    data=fp,
                    file_name="trackedFaces.zip",
                    mime="application/zip"
                )

                if btn:
                    # Remove the temporary video file
                    os.remove(temp_filename)
                    st.toast("Downloaded the File succesfully",icon="✅")
                    time.sleep(5)
                    st.stop()



if __name__=="__main__":
    st.header("Face Tracking using Face_recognition library")
    st.divider()
    main()




