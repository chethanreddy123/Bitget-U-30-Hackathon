
def getPromptJobRecommendation(jobTitle, jobDescription):

    jobRecommendataion = f'''
    Act like an expert content writer:

    A Job title and Job Description are given below. Now re-write the Job Description 
    in 3 different formats according to the Job title, make sure that the job description is very appealing
    and try to include as many synonyms as possible for the skills and job description
    is properly structured and formatted. The job description should be atleast 500 words long.

    Job Title: {jobTitle}

    Job Description: {jobDescription}

    Note: The output should be in a list of JSON format like given below. Give 5 responses in different formats.

    Output:
    [
    {{
        "jobTitle": "the title of the job",
        "jobDescription": "the refined description of the job",
    }},
     {{
        "jobTitle": "the title of the job",
        "jobDescription": "the refined description of the job",
    }}
    ]
    

    Hints for Writing Job Descriptions:

    Job descriptions should be prepared in a manner that all components are accurately stated to create a clear understanding of the role. Here are some hints to assist you in the process:
    1. Write in a concise, direct style.
    2. Always use the simpler word rather than the complicated one; keeping sentence structure as simple as possible. It will cut verbiage, shorten your description, and enhance understanding.
    3. Use descriptive action verbs in the present tense (for example: writes, operates, or performs).
    4. Avoid abbreviations and acronyms. Other people reading the position description may not be familiar with them. If abbreviations and acronyms are necessary, define them the first time you use them.
    5. Don't use ambiguous terms. If you use terms such as “assists, handles, and performs,” describe “how” the position assists, handles, or performs. Using the word “by” and then detailing the processes, tasks, or operations performed will usually clarify the ambiguity.
    6. Avoid gender-specific language, such as, “He manages,” “She is responsible for.”
    7. Focus on essential activities; omit trivial duties and occasional tasks.
    8. Avoid references to other employee’s names, instead, refer to the job title or department.
    9. Only include assigned duties today. Do not include potential future duties and eliminate any duties no longer required.
    '''

    return jobRecommendataion



def getPromptJobEnhancement(jobTitle, jobDescription):

    jobEnhancement = f'''
    Act like an expert content writer and Human Resource:

    For the given below Job Title and Job Description, you have to 
    give points to enhance the Job Description. 

    Job Title: {jobTitle}

    Job Description: {jobDescription}

    Note: The output should be in a JSON format like given below.

    Output:
    {{
        "enhancementPointers": [all the pointers in a list],
    }}

    Use the below hints to give enhancement pointers for the Job Description:

    Job descriptions should be prepared in a manner that all components are accurately stated to create a clear understanding of the role. Here are some hints to assist you in the process:
    1. Write in a concise, direct style.
    2. Always use the simpler word rather than the complicated one; keeping sentence structure as simple as possible. It will cut verbiage, shorten your description, and enhance understanding.
    3. Use descriptive action verbs in the present tense (for example: writes, operates, or performs).
    4. Avoid abbreviations and acronyms. Other people reading the position description may not be familiar with them. If abbreviations and acronyms are necessary, define them the first time you use them.
    5. Don't use ambiguous terms. If you use terms such as “assists, handles, and performs,” describe “how” the position assists, handles, or performs. Using the word “by” and then detailing the processes, tasks, or operations performed will usually clarify the ambiguity.
    6. Avoid gender-specific language, such as, “He manages,” “She is responsible for.”
    7. Focus on essential activities; omit trivial duties and occasional tasks.
    8. Avoid references to other employee’s names, instead, refer to the job title or department.
    9. Only include assigned duties today. Do not include potential future duties and eliminate any duties no longer required.

    '''

    return jobEnhancement



def keywordRecommendation(keywordsRegex, jobTitle, jobDescription):
    keywordsPrompt = f'''
    Act like a keyword analyzer for job description:

    Give below is a job title, job description, and list of 
    keywords extracted from the job description using regex 
    and NLP models, now you have to clean the keywords list 
    properly which are more relevant to the job description 
    and suggestion some more impactful keywords that can be 
    included in the job description:

    Job Title: {jobTitle}

    Job Description: {jobDescription}

    Keywords Extracted from NLP models and regex: {keywordsRegex}

    Note: The output should be in a JSON format like given below.

    Output:
        {{
            "cleanKeywords": [all cleaned keywords from give keywords list],
            "recommendedKeywords": [newly recommended keywords]
        }}

    '''
    
    return keywordsPrompt


def getScorePrompt(jobDescription, resumeContent):
    scorePrompt = f'''Act like an expert resume analyzer:
    For the given below job description and resume content give the 
    scores for all the pointers given below along with their average,

    Pointers: 
        1. Skills
        2. Education
        3. Recognition and Achievements
        4. experience
        5. Career progression
        6. Formatting Style

    Job Description: {jobDescription}


    Resume Content: {resumeContent}

    Note: Output Should be a json object like given below, don't give a null output.

    Output:
    {{
        "skillsScore": score for relevant skills out of 100,
        "education": score for relevant education out of 100,
        "recognitionAchievements" : score for Recognition and Achievements out of 100 need not be relavent to the job description,
        "experience": score for relevant experience out of 100,
        "careerProgression" : score for career progression out of 100,
        "formattingStyle" : score for formatting style and writing out of 100,
    }}

    Example Output:
    {{
        "skillsScore": 90.23,
        "education": 80.45,
        "recognitionAchievements": 80.45,
        "experience": 80.56,
        "careerProgression": 80.67,
        "formattingStyle": 80.65,
        "averageScore": 82.5
    }}
    '''



    return scorePrompt


def getPromptJobDescription(jobTitle):
    prompt = f'''
    Act like an expert content writer (Human Resource)):
    Creating a job description for the given job title using the addtional pointers given below, make sure the job description is atleast500 words long.

    Job Title: {jobTitle}

    Use the below hints to give enhancement pointers for the Job Description:

    Job descriptions should be prepared in a manner that all components are accurately stated to create a clear understanding of the role. Here are some hints to assist you in the process:
    1. Write in a concise, direct style.
    2. Always use the simpler word rather than the complicated one; keeping sentence structure as simple as possible. It will cut verbiage, shorten your description, and enhance understanding.
    3. Use descriptive action verbs in the present tense (for example: writes, operates, or performs).
    4. Avoid abbreviations and acronyms. Other people reading the position description may not be familiar with them. If abbreviations and acronyms are necessary, define them the first time you use them.
    5. Don't use ambiguous terms. If you use terms such as “assists, handles, and performs,” describe “how” the position assists, handles, or performs. Using the word “by” and then detailing the processes, tasks, or operations performed will usually clarify the ambiguity.
    6. Avoid gender-specific language, such as, “He manages,” “She is responsible for.”
    7. Focus on essential activities; omit trivial duties and occasional tasks.
    8. Avoid references to other employee’s names, instead, refer to the job title or department.
    9. Only include assigned duties today. Do not include potential future duties and eliminate any duties no longer required.

    Note: Make sure that the output is structured correctely with no bold words (*) and just having simple text.

    '''

    return prompt

#  for email
def emailPrompt(resumeContent , candId):
    prompt = f'''Act like a question setter for interviews:

    Create a set of 5 video recording questions for the candidate 
    based on his resume questions given below, and make sure that 
    the response is not more than for that question. Add note that 
    the candidate can record not more 1 min each repsonse. Include
    the candidate Id : {candId} in the email. The email sender is hAIr.

    Note: After creating the questions create a short email congratulating 
    the candidate for selecting a video interview and ask him to record the video
    for the respective question and upload it in the given link and add start 
    the email with the "dear candidate". Generate on the questions alone.

    Link: https://hair-axis.netlify.app/firstRoundInterview
    

    Resume Content: {resumeContent}
    '''

    return prompt


def getPromptVidoeAnalysis(EmailQuestionary, VideoAnalysisResults, AudioAnalysisResults, ResumeContent):
    prompt = f'''Act like the expert interviewer:
        Give below is a set of questions sent to a candidate through email, 
        now I have already done an intense analysis of the audio and videos 
        of the candidate, now give a score out of 100 without any bias make sure 
        that the scores are with decimal points as well.

        Email Questionary: {EmailQuestionary}

        Vidoe Analysis Results: {VideoAnalysisResults}

        Audio Analysis Results: {AudioAnalysisResults}

        Resume Content: {ResumeContent}

        Output: Make sure that the output is JSON like give below:

        {{
            "output": "the final score"
        }}

        FinalScore = (VideoAnalysisResults + AudioAnalysisResults) / 2

    '''

    return prompt


def getPromptForGaps(ResumeContent , JobDescription):

    prompt = f'''Act as an expert HR and Trainer:

    Given the resume content of a candidate and a job description provided below, 
    your task is to identify any gaps in the candidate's skills and experiences 
    compared to the job requirements. Additionally, suggest a list of specific 
    pointers to help the candidate enhance their resume content and align it with 
    the job description. 

    Furthermore, propose relevant online courses and HR training programs 
    that can assist the candidate in acquiring the necessary skills and knowledge 
    for this role.

    Resume Content: {ResumeContent}

    Job Description: {JobDescription}

    Additional Information:
    Please provide detailed and actionable recommendations for both the 
    resume content and job description. Consider the candidate's existing 
    skills and experiences, and focus on areas that require improvement or 
    further development. Additionally, suggest specific online courses and 
    HR training programs that can benefit the candidate in meeting the job 
    requirements.

    Final Output:

    Note: Put the output in the following JSON structure and give the output.

    {{
        "ResumeGaps": [List of gaps in the resume from job description(max 5 strings) ],
        "HR_Training_Programs" :  [List of methodological training programs related to non-tech (max 5 strings)] ,
        "Courses" : [List of courses name (max 5 strings)]
    }}
    '''


    return prompt

def convertingProperJson(rawText):
    prompt = f"""Act like expert programmer and convert the given text into proper json format:

    Raw Text: {rawText}
    """

    return prompt

def promptSummary(ResumeContent, JobDescription):
    prompt = f'''Act like an expert content writer:

    Given the resume content of a candidate and a job description provided below, 
    your task is to create a summary in about 100 words of the candidate's skills and experiences 
    compared to the job requirements.

    Resume Content: {ResumeContent}

    Job Description: {JobDescription}

    Additional Information:
    Please provide a summary of the candidate's skills and experiences compared 
    to the job requirements. Consider the candidate's existing skills and experiences, 
    and focus on areas that require improvement or further development. 
    '''
    return prompt
