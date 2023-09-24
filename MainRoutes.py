from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from loguru import logger
import prompts
import google.generativeai as palm
from dotenv import load_dotenv
import os
import shutil
from fastapi import FastAPI, File, UploadFile
import re
import JobEvaluator
from ResumeMatcher.scripts.JobDescriptionProcessor import JobDescriptionProcessor
from ResumeMatcher.scripts.ResumeProcessor import ResumeProcessor
from pymongo.mongo_client import MongoClient
import json
import random as rd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl
import csv
import random
from email.message import EmailMessage
from pytube import YouTube
import os
import cv2
import numpy as np
from nltk.corpus import stopwords
from uuid import uuid1
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pytube import YouTube
import assemblyai as aai

load_dotenv()

aai.settings.api_key = os.environ.get("ASM_KEY")
# achethanreddy1921@gmail.com, xkdqwfgstwndbdtp


##### MongoDB #####

Key_Mongo_Cloud = os.environ.get("MONGO_KEY")
JobDescriptionData = MongoClient(Key_Mongo_Cloud)['HairBackEndData']['JobDescription']
ResumeData = MongoClient(Key_Mongo_Cloud)['HairBackEndData']['ResumeData']

#### Additional functions ####


def delete_all_files_in_directory(directory_path):
    try:
        # List all files in the directory
        file_names = os.listdir(directory_path)

        # Delete each file in the directory
        for file_name in file_names:
            file_path = os.path.join(directory_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        
        print("All files deleted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_average_score(data):
    total_score = 0
    num_keys = len(data)

    for key, score in data.items():
        total_score += score

    average_score = total_score / num_keys
    return average_score

def preprocess_json(json_data):
    # Convert JSON to a set of lowercase words (to remove duplicates and ignore case)
    words_set = set(word.lower() for word in json_data.get("extracted_keywords", []))
    return words_set

def calculate_jaccard_similarity(job_description, resume):
    # Preprocess the job description and resume JSON data
    job_desc_set = preprocess_json(job_description)
    resume_set = preprocess_json(resume)

    # Calculate the Jaccard similarity
    intersection_size = len(job_desc_set.intersection(resume_set))
    union_size = len(job_desc_set.union(resume_set))
    jaccard_similarity = intersection_size / union_size

    # Convert Jaccard similarity to a 0-100 range
    jaccard_similarity_0_to_100 = jaccard_similarity * 100

    return jaccard_similarity_0_to_100

def read_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def perform_emotion_analysis(video_path):
    face_classifier = cv2.CascadeClassifier('modelsVideos/haarcascade_frontalface_default.xml')
    classifier = load_model('modelsVideos/Emotion_little_vgg.h5')

    class EmotionHandler:
        def __init__(self):
            self.interrupt = False
            self.status = {'label': {'Angry': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0},
                           'processed': 0,
                           'frames': 0}
            
    def check_video_file(file_path):
        _path, file_name = os.path.split(file_path)
        split_list = file_name.split('.')
        extension = split_list[-1] if len(split_list) > 1 else None

        if extension == 'webm':
            new_file_path = os.path.join(_path, f'{uuid1()}.mp4')
            command = f'./support/ffmpeg -v quiet -stats -hwaccel qsv -i {file_path} -r 15 -c:v h264_qsv -preset faster {new_file_path}'
            os.system(command)
            return new_file_path
        return file_path
            
    def emotion_analyse(file_path, handler):
        file_path = check_video_file(file_path)
        handler.status['label'] = {'Angry': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
        handler.status['processed'] = 0
        cap = cv2.VideoCapture(file_path)
        handler.status['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        while not handler.interrupt:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = classifier.predict(roi)[0]
                    label = class_labels[preds.argmax()]
                    handler.status['label'][label] += 1
            handler.status['processed'] += 1

        cap.release()
        cv2.destroyAllWindows()
        
        total_processed_frames = handler.status['processed']
        for emotion, count in handler.status['label'].items():
            handler.status['label'][emotion] = (count / total_processed_frames) * 100
        
        return handler.status
    
    handler = EmotionHandler()
    result = emotion_analyse(video_path, handler)
    return result

def Download(link):
        youtubeObject = YouTube(link)
        youtubeObject = youtubeObject.streams.get_highest_resolution()
        try:
            youtubeObject.download(output_path='downloadedVideos')
        except:
            print("An error has occurred")
        print("Download is completed successfully")

palm.configure(api_key=os.environ.get("PALM_API_KEY"))
def generateTextWithPalm(prompt):
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        max_output_tokens=10000 ,
    )
    return completion.result

def clean_string(raw_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', raw_string)
    return cleaned_string

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###### ----- Job Description Recommendation ----- ######
@app.get("/healthCheck")
async def loginCheck():
    return {"status": True}

@app.post("/jobDescriptionRecommendation")
async def jobDescriptionRecommendation(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for job description recommendations")
    jobTitle = req_info["jobTitle"]
    jobDescription = req_info["jobDescription"]

    promptRecommendation = prompts.getPromptJobRecommendation(jobTitle, jobDescription)
    results = generateTextWithPalm(promptRecommendation)

    logger.info("job description recommendations review completed")
    # Use regex to extract job titles and job descriptions
    pattern = r'"jobTitle": "(.*?)",\n\s*"jobDescription": "(.*?)"'
    matches = re.findall(pattern, results, re.DOTALL)

    logger.info("job description recommendataion completed")

    # Create a list of dictionaries containing job titles and job descriptions
    jobList = [{"jobTitle": title, "jobDescription": description} for title, description in matches]

    jobRecommendations = {
        "jobRecommendations" : jobList
    }

    return {"status": jobRecommendations}


@app.post("/jobDescriptionTextEvaluation")
async def jobDescriptionTextEvaluation(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    print(req_info)
    logger.info("recived request for job description evaluation")

    jobTitle = req_info["jobTitle"]
    jobDescription = req_info["jobDescription"]

    promptEnhancement = prompts.getPromptJobEnhancement(jobTitle, jobDescription)
    results = generateTextWithPalm(promptEnhancement)
    json_string = results

    # Use regex to extract the content of the "enhancementPointers" list
    pattern = r'"enhancementPointers": \[([\s\S]*?)\]'
    matches = re.search(pattern, json_string)

    if matches:
        pointers_list_string = matches.group(1)
        enhancement_pointers = re.findall(r'"(.*?)"', pointers_list_string)
    else:
        enhancement_pointers = []   

    pointerEnhancement = {"enhancement_pointers" : enhancement_pointers}
    keyWordsModel = JobEvaluator.DataExtractor(jobDescription)
    keywordsRegex = keyWordsModel.extract_keywords()
    logger.info("NLP and regex keywords are imported")

    promptRecommendationKeywords = prompts.keywordRecommendation(keywordsRegex , jobTitle, jobDescription)
    results = generateTextWithPalm(promptRecommendationKeywords)

    json_string = results

    logger.info(json_string)

    # Use regex to extract the content of the "cleanKeywords" list
    clean_keywords_matches = re.search(r'"cleanKeywords": \[([\s\S]*?)\]', json_string)
    if clean_keywords_matches:
        clean_keywords_list_string = clean_keywords_matches.group(1)
        clean_keywords = re.findall(r'"(.*?)"', clean_keywords_list_string)

    # Use regex to extract the content of the "recommendedKeywords" list
    recommended_keywords_matches = re.search(r'"recommendedKeywords": \[([\s\S]*?)\]', json_string)
    if recommended_keywords_matches:
        recommended_keywords_list_string = recommended_keywords_matches.group(1)
        recommended_keywords = re.findall(r'"(.*?)"', recommended_keywords_list_string)

    logger.info("cleaned the keywords and recommendations for job description evaluation")

    
    finalResults = {
        "cleanKeywords": clean_keywords,
        "recommendedKeywords": recommended_keywords,
        "enhancementPointers" : pointerEnhancement['enhancement_pointers']
    }

    logger.info("Completed job description evaluation")

    return finalResults

@app.post("/processJobDescription")
async def processJobDescription(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for job description processing")
    model = JobDescriptionProcessor(req_info['jobDescription'])
    status = model.process()
    if status == True:
        currData = read_json_file("JobDescription.json")
        currData['JobId'] = "AXIS" + str(rd.randint(100000, 999999))
        currData['JobTitle'] = req_info['jobTitle']
        Check = JobDescriptionData.insert_one(currData)
        if Check.acknowledged == True:
            return {"status": "success"}
        else:
            return {"status": "failed"}
    else:
        return {"status": "failed"}


@app.get("/getAllJobs")
async def getAllJobs():
    logger.info("recived request for all jobs")
    currData = list(JobDescriptionData.find({}))
    for i in range(len(currData)):
        del currData[i]['_id']
    return {"result" : currData}

@app.post("/getJobDescription")
async def getJobDescription(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)

    logger.info("recived request for job description")

    jobTitle = req_info["jobTitle"]
    PromptDescription = prompts.getPromptJobDescription(jobTitle)
    result = generateTextWithPalm(PromptDescription)

    return {"jobDescription" : result}



###### ----- Resume Recommendation ----- ######

@app.post("/analyzeResume")
async def analyzeResume(pdfFile: UploadFile = File(...),
                        text: str = Form(...)):

    # Save the uploaded PDF to the server with the name "currResume.pdf"
    with open("currResume.pdf", "wb") as f:
        f.write(pdfFile.file.read())

    logger.info(f"here is jobID: {text}")
    logger.info("received request for resume analysis")

    model = ResumeProcessor("currResume.pdf")
    model.process()
    currResumeData = read_json_file("ResumeEntites.json")
    currJobData = dict(JobDescriptionData.find_one({"JobId": text}))

    # Calculate the similarity score between the resume and the job description:
    jaccard_similarity = calculate_jaccard_similarity(currJobData, currResumeData)
    logger.info(f"Jaccard Similarity: {jaccard_similarity:.2f}")
    logger.info("now analysing resume with LLMs")

    scorePrompt = prompts.getScorePrompt( clean_string(currJobData['resume_data']) , clean_string(currResumeData['resume_data']) )
    logger.info(scorePrompt)
    results = generateTextWithPalm(scorePrompt)
    logger.info("LLMs analysis completed")
    logger.info(results)
    scoreCard = json.loads(results)


    logger.info("Calculating the Score Card")
    average = calculate_average_score(scoreCard)
    logger.info(f"Score Card: {average}")
    finalScore = (0.3 * average) + (0.7 * jaccard_similarity * 100)
    logger.info(f"Final Score: {finalScore:.2f}")
    logger.info("resume analysis completed, data pushing to mongoDB")
    currResumeData['score'] = finalScore
    currResumeData['jobId'] = text
    print(currResumeData)
    del currResumeData['pos_frequencies']

    # I have scoreCard and Jackard Similarity => have to combine this make once score
    # 70% priority to scoreCard and 30% to Jackard Similarity
    # # Push the results to MongoDB

    check = ResumeData.insert_one(currResumeData)

    if  check.acknowledged == True:
        logger.info("resume analysis completed")
        return {"status": "success"}
    else:   
        return {"status": "failed"}
    
@app.post("/startCVMatching")
async def startCVMatching(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)

    logger.info("received request for resume recommendation")
    cursor = ResumeData.find({"jobId" : req_info['jobId']}, {'_id' : 0 ,'unique_id': 1, 'emails': 1, 'score': 1 , 'jobId' : req_info['jobId']}).sort('score', -1)
    res = list(cursor)

    return {"result" : res}
    

@app.post("/sendEmail")
async def sendEmail(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for sending email")
    topIds = req_info['sendEmails']

    emailSender = 'aioverflow.ml@gmail.com'
    emailPassword = "iyfngcdhgfcbkufv"

    for cand in  topIds:
        subject = "Congratulations on Advancing to the Video Interview Stage!"
        resumeContent = ResumeData.find_one({"unique_id" : cand['unique_id']})
        resumeContent = dict(resumeContent)
        resumeContent = resumeContent['resume_data']
        body = generateTextWithPalm(prompts.emailPrompt(resumeContent , candId=cand['unique_id']))
        logger.info(f"email sent to {cand['emails'][0]}")
        em = EmailMessage()
        em['From'] = "aioverflow.ml@gmail.com"
        em['To'] = cand['emails'][0]
        em['Subject'] = subject 
        em.set_content(body)
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('smtp.gmail.com' , 465, context=context) as smtp:
            smtp.login(emailSender , emailPassword)
            smtp.sendmail(emailSender, cand['emails'], em.as_string())
        
        ResumeData.update_one({"unique_id" : cand['unique_id']}, {"$set" : {"emailBody" : body}})
    return {"status" : "success"}

@app.post("/getEmailBody")
async def getEmailBody(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for video analysis")

    candId = req_info['unique_id']
    completeResume = ResumeData.find_one({"unique_id" : candId})

    return completeResume['emailBody']
        
@app.post("/videoAnalysis")
async def videoAnalysis(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for video analysis")

    videoLinks = [i for i in req_info['videoLinks'].values()]
    candId = req_info['unique_id']
    logger.info(videoLinks)

    for link in videoLinks:
        Download(link)

    file_names = os.listdir('downloadedVideos')
    OnlyVideoAnalysis = []
    for videos in file_names:
        OnlyVideoAnalysis.append(perform_emotion_analysis(f'downloadedVideos/{videos}'))
    
    logger.info("video only analysis completed")


    for audio in videoLinks:
        yt=YouTube(audio)
        t=yt.streams.filter(only_audio=True).all()
        t[0].download("downloadedAudio")

    extractedText = []
    file_names = os.listdir('downloadedAudio')

    for audio in file_names:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(f'downloadedAudio/{audio}')
        extractedText.append(transcript.text)

    logger.info("audio only analysis completed")

    completeResume = ResumeData.find_one({"unique_id" : candId})


    actualPrompt = prompts.getPromptVidoeAnalysis(completeResume['emailBody'], OnlyVideoAnalysis, extractedText, completeResume['clean_data'])
    results = generateTextWithPalm(actualPrompt)

    logger.info("video analysis completed")

   
    folder_path = 'downloadedAudio'
    delete_all_files_in_directory(folder_path)

    folder_path = 'downloadedVideos'
    delete_all_files_in_directory(folder_path)

    data = json.loads(results)

    res =  {
            "result" : data,
            "videoAnalysis" : OnlyVideoAnalysis,
            "audioAnalysis" : extractedText,
        }
    
    print(res)

    ResumeScore =  completeResume['score'] / 1000
    VideoAnswers = data['output'] / 100

        # Define weights for each component (you can adjust these based on importance)
    weight_resume = 0.7
    weight_video = 0.3

    # Calculate the final score
    final_score = (weight_resume * ResumeScore + weight_video * VideoAnswers) * 100

            


    ResumeData.update_one({"unique_id" : candId}, {"$set" : {"videoAnalysis" : res , "finalScore" : final_score}})
    
    return res


@app.get("/allCandidates")
async def allCandidates():
    currData = list(ResumeData.find({}))
    for i in currData:
        del i['_id']
    return currData

@app.post("/getTopCandidates")
async def getTopCandidates(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    logger.info("recived request for top x candidates")

    logger.info(req_info)
    logger.info(type(req_info['number']))

    jobId = req_info['jobId']
    number =  req_info['number']
    currData = list(ResumeData.find({"jobId" : jobId}).sort('score', -1).limit(number))

    FinalResult = {}

    # ResumeScores:
    score = [i['score'] for i in currData]
    emails = [i['emails'][0] for i in currData]
    ResumeScores = []
    for i, j in zip(emails, score):
        ResumeScores.append({"email" : i , "score" : j})
    FinalResult['ResumeScores'] = ResumeScores

    # VideoScores:
    VideoAnlytics = []

    for i in currData:
        cData = {
            "email" : i['emails'][0],
            "Angry" : sum([j['label']['Angry'] for j in i['videoAnalysis']['videoAnalysis']])/len([j['label']['Angry'] for j in i['videoAnalysis']['videoAnalysis']]),
            "Happy" : sum([j['label']['Happy'] for j in i['videoAnalysis']['videoAnalysis']])/len([j['label']['Happy'] for j in i['videoAnalysis']['videoAnalysis']]),
            "Neutral" : sum([j['label']['Neutral'] for j in i['videoAnalysis']['videoAnalysis']])/len([j['label']['Neutral'] for j in i['videoAnalysis']['videoAnalysis']]),
            "Sad" : sum([j['label']['Sad'] for j in i['videoAnalysis']['videoAnalysis']])/len([j['label']['Sad'] for j in i['videoAnalysis']['videoAnalysis']]),
            "Surprise" : sum([j['label']['Surprise'] for j in i['videoAnalysis']['videoAnalysis']])/len([j['label']['Surprise'] for j in i['videoAnalysis']['videoAnalysis']])
        }
        VideoAnlytics.append(cData)
    FinalResult['VideoAnlytics'] = VideoAnlytics

    # FinalScores:
    FinalScores = []

    for i in currData:
        cData = {
            "email" : i['emails'][0],
            "score" : i['finalScore']
        }
        FinalScores.append(cData)
    FinalResult['FinalScores'] = FinalScores


    return FinalResult

@app.post("/getGaps")
async def getGaps(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    unique_id = req_info['unique_id']
    logger.info("recived request for resume gaps")
    currData = ResumeData.find_one( {"unique_id" : unique_id})

    JobData = JobDescriptionData.find_one({"JobId" : currData['jobId']})    
    PromptGaps = prompts.getPromptForGaps(clean_string( currData['clean_data']) , JobData  )

    results = generateTextWithPalm(PromptGaps)


    results = results.replace('json' , '').replace('```' , "").replace("\n" , "")

    results = generateTextWithPalm(prompts.convertingProperJson(results))
    results = results.replace('json' , '').replace('```' , "").replace("\n" , "")


    # Remove extra spaces and newlines for valid JSON format
    cleaned_string = ' '.join(results.split())

    # Convert string to a dictionary
    resume_data_gaps = json.loads(cleaned_string)
    logger.info(resume_data_gaps)

    NewList = []
    for i in resume_data_gaps['Courses']:
        name = i
        link = f'https://www.coursera.org/search?query={name.replace(" " , "%20")}'
        NewList.append({"CourseName" : name , "CourseLink" : link})
    
    resume_data_gaps['Courses'] = NewList


    ResumeData.update_one( {"unique_id" : unique_id} , {"$set" : {"ResumeGaps" : resume_data_gaps}})

    logger.info("Update the resume gaps")

    return  resume_data_gaps


@app.post("/getSummary")
async def getSummary(info : Request):
    req_info = await info.json()
    req_info = dict(req_info)
    unique_id = req_info['unique_id']
    logger.info("recived request for resume summary")
    currData = ResumeData.find_one( {"unique_id" : unique_id})

    JobData = JobDescriptionData.find_one({"JobId" : currData['jobId']})    
    PromptSummary = prompts.promptSummary(clean_string( currData['clean_data']) , JobData  )

    results = generateTextWithPalm(PromptSummary)

    logger.info(results)

    ResumeData.update_one( {"unique_id" : unique_id} , {"$set" : {"ResumeSummary" : results}})

    logger.info("Update the resume summary")

    return  results


@app.post("/login/hr")
async def login(info : Request):
    req_info = await info.json() 
    req_info = dict(req_info)
    logger.info("recived request for login")
    if req_info['username'] == "test" and req_info['password'] == "test":
        return {"status" : "success"}
    else:
        return {"status" : "failed"}
    

@app.post("/login/candidate")
async def login(info : Request):
    req_info = await info.json() 
    req_info = dict(req_info)
    logger.info("recived request for login")
    check = ResumeData.find_one({"unique_id" : req_info['unique_id']})
    logger.info(check)
    if check == None or req_info['unique_id'] != req_info['password']:
        return {"status" : "failed"}
    else:
        return {"status" : "success"}
    
