import io
import os
import pandas
import json
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cv2
import numpy as np
import scipy
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import scipy.cluster
import binascii
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
from nltk.stem import WordNetLemmatizer 
from emoji import UNICODE_EMOJI
import dill
from datetime import datetime as dt

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

working_directory='F:/ObjectDetection'
ctr_video_file = 'F:/ObjectDetection/CTR3and7_WNO.xlsx'

# The name of the folder containing images to annotate
img_directory=working_directory+'WNO/Thumbnails/'
output_directory=working_directory+'WNO/Outputs/'
responses_folder = output_directory+'Responses/'

with open(working_directory+'api_key.json','r') as f:
    api_key=json.load(f)['api_key']

credentials_json=working_directory+'service_account_key.json'

#Instantiates a client
credentials = service_account.Credentials.from_service_account_file(credentials_json)
client = vision.ImageAnnotatorClient(credentials=credentials)
#
colors_dict = {"Tone 3":(206, 142, 113),"Tone 2":(198, 136, 99),"Tone 1":(233, 200, 188)}

punctuations_tracked = ['?', '!','&',',']

face_detection_confidence_threshold = 0.49
web_detection_threshold = 1.5
logo_detection_threshold = 0
object_detection_threshold = 0.49
round_to_digits = 2

likelihood_name = {'UNKNOWN':1, 'VERY_UNLIKELY':2, 'UNLIKELY':4, 'POSSIBLE':3,
                   'LIKELY':4, 'VERY_LIKELY':5}

def get_image(img_path):
    '''Loads the image into memory'''
    file_name = os.path.abspath(img_path)
    
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    return image

def web_detection(img_path):
    '''Performs web detection on image file''' 
    
    image=get_image(img_path)
    response = client.web_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    annotations = response.web_detection
    return annotations

def label_detection(img_path):
    '''Performs label detection on the image file'''
    
    image=get_image(img_path)
    response = client.label_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    labels = response.label_annotations
    print('Label{} Found: {}'.format('' if len(labels) <= 1 else 's',len(labels)))
    
    return labels

def face_detection(img_path):
    '''Performs face detection on the image file'''
    
    image=get_image(img_path)
    response = client.face_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    faces = response.face_annotations
    print('Face{} found: {}'.format('' if len(faces) <= 1 else 's', len(faces)))
    
    return faces
    

def object_localization(img_path):    
    ''' Performs object detection on the image file'''
    
    image=get_image(img_path)
    response= client.object_localization(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    objects = response.localized_object_annotations
    print('Object{} Found: {}'.format('' if len(objects) <= 1 else 's',len(objects)))
    
    return objects

def detect_text(img_path):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    image=get_image(img_path)
    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    texts = response.text_annotations
    
    return texts


def detect_logos(img_path):
    """Detects logos in the file."""
    
    image=get_image(img_path)
    response = client.logo_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    logos = response.logo_annotations
    print('Logo{} Found: {}'.format('' if len(logos) <= 1 else 's',len(logos)))
    
    return logos

def get_dominant_color(im,remove_black=False):

    ar = np.asarray(im)
    ar = ar.reshape(np.product(ar.shape[:2]), ar.shape[2]).astype(float)
    if remove_black:
        for i,arr in enumerate(ar):  
            if sum(arr)<15: 
                ar= np.delete(ar,i,0)
    NUM_CLUSTERS = 3
    
    codes, _ = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, _ = scipy.cluster.vq.vq(ar, codes)
    counts, _ = np.histogram(vecs, len(codes))
    
    index_max = np.argmax(counts)
    peak = codes[index_max]
    color = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    print('most frequent color is {}'.format(color))
    
    return peak

def get_quadrant(im,points):
    x, y = im.size
    quadrants_present_in = list()
    t1 , t2 = int(x/2), int(y/2)
    for point in points:
        p1, p2 = point       
        if 0<p1<t1 and 0<p2<t2 : quadrants_present_in.append('Q1')
        if 0<p1<t1 and t2<p2<y : quadrants_present_in.append('Q2')
        if t1<p1<x and t2<p2<y : quadrants_present_in.append('Q3')
        if t1<p1<x and 0<p2<t2 : quadrants_present_in.append('Q4')
    quadrants_present_in = list(dict.fromkeys(quadrants_present_in))
    return ','.join(quadrants_present_in)

denormalize_vertices = lambda x,shape : x*shape
format_scores= lambda x : str(format(x*100, '.3f')) + '%' 
get_area = lambda box : (box[1][0]-box[0][0])*(box[2][1]-box[1][1])

def get_percent_covered(im , box, is_box = False):
    if is_box : 
        image_box = im
    else:
        image_box = [(0,0),(im.size[0],0),im.size,(0,im.size[1])]
    area1 = get_area(image_box)
    area2 = get_area(box)
    return round((area2/area1)*100)

def get_color_name(peak):
    distance={}
    color_rgb1=sRGBColor(peak[0], peak[1], peak[2])
    color1_lab = convert_color(color_rgb1, LabColor)
     
    for key,value in colors_dict.items():
        
        color_rgb2=sRGBColor(colors_dict[key][0], colors_dict[key][1], colors_dict[key][2])
        color2_lab = convert_color(color_rgb2, LabColor)
        distance[delta_e_cie2000(color1_lab, color2_lab)]= key

    return distance[min(distance.keys())]

def get_chyron_band(img_path):
    
    im = cv2.imread(img_path)
    
    y=int(len(im)/4)*3
    crop = im[y:,:]
    
    im_bgr=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(im_bgr, cv2.COLOR_RGB2GRAY)
    
    edged=cv2.Canny(gray, 127, 255,apertureSize = 3)
#    contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#    diag=cv2.drawContours(im,contours, -1, (0,255,0),2)
    try:
        cv2.HoughLines(edged,1,np.pi/180, 200)[0]
        return 1
    except:
        return 0

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:          
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = [word if tag is None else lemmatizer.lemmatize(word, tag) for word, tag in wn_tagged]
  return " ".join(res_words)

 
def clean_caption(text):
    
    text=text.lower()
    text=re.sub('[!"$%&\'()*+,-./:;<=>?[\\]^`{|}~•]+','', text)
    text = re.sub("[^a-zA-Z0-9,@#//:_ -]","",text)
    
    text_words=text.split()
    sentence=' '.join([word for word in text_words if word not in stop_words])
    
    sentence= lemmatize_sentence(sentence)

    return sentence

def get_emotion(face):
    joy_score = likelihood_name[face['joyLikelihood']]
    sorrow_score = likelihood_name[face['sorrowLikelihood']]
    surprise_score = likelihood_name[face['surpriseLikelihood']]
    anger_score = likelihood_name[face['angerLikelihood']]
    emotions = {0:'Joy' , 1:'Sorrow', 2:'Surprise', 3:'Anger'}
    emotions_band = [joy_score, sorrow_score, surprise_score, anger_score]
    emotion_dict = {}
    for n,i in enumerate(emotions_band):
        if emotion_dict.get(i,''):
            emotion_dict[i] = 'Neutral'
        else:
            emotion_dict[i] = emotions[n]
    
    return emotion_dict[max(emotions_band)]

analyzer = SentimentIntensityAnalyzer()

df = pandas.read_excel(ctr_video_file)

df['CTR day3'] *= 100 
df['CTR day7'] *= 100
df['date_published'] = df['date_published'].apply(lambda x : x.strftime('%Y-%m-%d')) 

stop_words=set(stopwords.words('english'))
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^`{|}~•'
lemmatizer = WordNetLemmatizer()


df2=pandas.DataFrame()

for index, row in df.iterrows():
    print('{}/{}'.format(index,len(df)))
    img_id = row['video_id']
    df2.loc[index,'video_id'] = img_id
    title = df.loc[index,'video_title_text']
    print('\n{}.jpg - {}'.format(img_id,title))
    img_path=img_directory+img_id+'.jpg'
    
    df.loc[index,'thumbnail_url']="https://i.ytimg.com/vi/"+img_id+"/maxresdefault.jpg"

    face_output_filename = output_directory+'Faces/'+img_id+'.jpg'
    objects_output_filename = output_directory+'Objects/'+img_id+'.jpg'
    text_output_filename = output_directory+'Text/'+img_id+'.jpg'
    logo_output_filename = output_directory+'Logo/'+img_id+'.jpg'
    
    sentiment = analyzer.polarity_scores(title)
    df.loc[index,'title_text_sentiment_net'] = round(sentiment['compound'],round_to_digits)
    df.loc[index,'title_text_sentiment_pn'] = round(sentiment['pos'] * sentiment['neg'],round_to_digits) 
    df.loc[index,'title_text_sentiment_pos'] = round(sentiment['pos'],round_to_digits)
    df.loc[index,'title_text_sentiment_neg'] = round(sentiment['neg'],round_to_digits)
    df.loc[index,'title_text_sentiment_neu'] = round(sentiment['neu'],round_to_digits)
    
    sentiment = analyzer.polarity_scores(title.upper())
    df.loc[index,'title_upper_text_sentiment_net'] = round(sentiment['compound'],round_to_digits)
    df.loc[index,'title_upper_text_sentiment_pn'] = round(sentiment['pos'] * sentiment['neg'],round_to_digits) 
    df.loc[index,'title_upper_text_sentiment_pos'] = round(sentiment['pos'],round_to_digits)
    df.loc[index,'title_upper_text_sentiment_neg'] = round(sentiment['neg'],round_to_digits)
    df.loc[index,'title_upper_text_sentiment_neu'] = round(sentiment['neu'],round_to_digits)

    df.loc[index,'punctuation_marks'] = sum([1 for i in punctuations_tracked if i in title])
    df.loc[index,'title_text_length'] = len(title)
    
    emoji_count = 0
    for emoji , description in UNICODE_EMOJI.items():
        if title.count(emoji) :
            description = description.replace(':','')
            df2.loc[index,'emoji_'+description] = title.count(emoji)
            emoji_count += title.count(emoji)
    
    df.loc[index,'emojis_count'] = emoji_count 
    
    sentence = clean_caption(title)
    for word in sentence.split(): df2.loc[index,word] = sentence.count(word)

    try:
        labels_detected = dill.load(file = open(responses_folder+'labels/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        labels_detected = [json.loads(MessageToJson(label)) for label in label_detection(img_path)]
        dill.dump(labels_detected, file = open(responses_folder+'labels/{}.pkl'.format(img_id),'wb'))
    
    for label in labels_detected: df2.loc[index,'label_'+label['description']] = 1 
          
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 50)

    try:
        faces_detected = dill.load(file = open(responses_folder+'faces/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        faces_detected = [json.loads(MessageToJson(face)) for face in face_detection(img_path)]
        dill.dump(faces_detected, file = open(responses_folder+'faces/{}.pkl'.format(img_id),'wb'))


    colors_detected= list()
    emotions =list()
    area_covered_face = 0
    eye_count = 0
    percent_covered = 0
    face_count =0
    for n,face in enumerate(faces_detected):
        if face['detectionConfidence'] < face_detection_confidence_threshold : continue
        try:
            fd_face_box = [(vertex['x'], vertex['y']) for vertex in face['fdBoundingPoly']['vertices']]
            full_face_box = [(vertex['x'], vertex['y']) for vertex in face['boundingPoly']['vertices']]
        except KeyError:
            continue
            
        draw.line(fd_face_box + [fd_face_box[0]], width=5, fill='#00ff00')
        draw.text((face['fdBoundingPoly']['vertices'][0]['x'],
                   (face['fdBoundingPoly']['vertices'][0]['y'] - 30)),
                  text= f'Face{n} : ' + format_scores(face['detectionConfidence']),
                  fill='#000000',font=font)
                  
        draw.line(full_face_box + [full_face_box[0]], width=5, fill='#00ff00')
        percent_covered += get_percent_covered(im,fd_face_box)
        area_covered_face += get_area(full_face_box)
        print('Face {} covers {} % of the image'.format(n,percent_covered))
        
        eye_area_covered_image = 0
        eye_area_covered_face = 0
        area_covered_eyes=0
        for eye in ['LEFT','RIGHT']:
            eye_landmarks = [f'{eye}_EYE_TOP_BOUNDARY',f'{eye}_EYE_RIGHT_CORNER',f'{eye}_EYE_BOTTOM_BOUNDARY',f'{eye}_EYE_LEFT_CORNER']                
            eye_box =[(landmark['position']['x'], landmark['position']['y']) for landmark in face['landmarks'] if landmark.get('type','') in eye_landmarks]
            draw.line(eye_box + [eye_box[0]], width=5, fill='#00ff00')
            eye_count +=1
            eye_area_covered_face += get_percent_covered(fd_face_box,eye_box,is_box =True)
            eye_area_covered_image += get_percent_covered(im,eye_box)
            area_covered_eyes +=  get_area(eye_box)
        
        left, top, right, bottom = (fd_face_box[0][0], fd_face_box[0][1] , fd_face_box[2][0] , fd_face_box[2][1])
        squeeze_width , squeeze_height = 0.01*im.size[0] , 0.01*im.size[1]
        im_crop = im.crop((left+squeeze_width, top+squeeze_height, right-squeeze_width, bottom-squeeze_height))        
        peak = get_dominant_color(im_crop)
        category_color = get_color_name(peak)
        colors_detected.append(category_color)
        draw.text((full_face_box[0][0],
                   full_face_box[0][1] - 30),
                  text= '#'+category_color,
                  fill='#000000',font=ImageFont.truetype('arial.ttf', 20))

        emotions.append(get_emotion(face))
        
        face_count +=1
    
    df.loc[index,'number_of_faces'] = face_count
    df.loc[index,'emotion_joy'] = emotions.count('Joy')
    df.loc[index,'emotion_sorrow'] = emotions.count('Sorrow')
    df.loc[index,'emotion_surprised'] = emotions.count('Surprise')
    df.loc[index,'emotion_anger'] = emotions.count('Anger')
    df.loc[index,'emotion_neutral'] = emotions.count('Neutral')
    df.loc[index,'Color_of_skin 1'] = colors_detected.count('Tone 1')
    df.loc[index,'Color_of_skin 2'] = colors_detected.count('Tone 2')
    df.loc[index,'Color_of_skin 3'] = colors_detected.count('Tone 3')
    df.loc[index,'area_covered_by_faces'] =  round(area_covered_face,0)
    df.loc[index,'area_covered_by_eyes'] =  round(area_covered_eyes,0)
    df.loc[index,'face_area_covered_image'] =  percent_covered
    df.loc[index,'eye_area_covered_face'] = eye_area_covered_face
    df.loc[index,'eye_area_covered_image'] = eye_area_covered_image
    df.loc[index,'eye_count'] = eye_count
    df.loc[index,'area_cvg_eye_over_face'] = round(area_covered_eyes/face_count, round_to_digits) if face_count else 0
    df.loc[index,'avg_area_covered_by_a_face'] = round(area_covered_face/face_count , round_to_digits) if face_count else 0
    df.loc[index,'avg_area_covered_by_an_eye'] = round(area_covered_eyes/eye_count , round_to_digits) if eye_count else 0
    im.save(face_output_filename)
    
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 50)
    
    try:
        text_detected = dill.load(file = open(responses_folder+'text/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        text_detected = [json.loads(MessageToJson(text)) for text in detect_text(img_path)]
        dill.dump(text_detected, file = open(responses_folder+'text/{}.pkl'.format(img_id),'wb'))   
    
    quadrants_present_in = list()
    area_covered = 0
    image_text =''
    for n,text in enumerate(text_detected):  
        if '\n' in text['description'] : continue
        try:
            text_box = ([(vertex['x'], vertex['y'])
                        for vertex in text['boundingPoly']['vertices']])
        except:
            continue
            
        draw.line(text_box + [text_box[0]], width=5, fill='#00ff00')
            
        draw.text((text_box[0][0],text_box[0][1] - 30),
                  text= text['description'],
                  fill='#000000',font=ImageFont.truetype('arial.ttf', 20))
    
        area_covered += get_area(text_box)
        percent_covered = get_percent_covered(im,text_box)
        quadrants_present_in.append(get_quadrant(im, text_box))
        image_text += text['description']+' '
    
    print('Text found on image : {}'.format(image_text))
    quadrants_present_in = [q for quadrant in list(dict.fromkeys(quadrants_present_in)) for q in quadrant.split(',')]
    quadrants_present_in = [q for q in list(dict.fromkeys(quadrants_present_in))]
    df.loc[index,'Image_text'] = image_text 
    df.loc[index,'text_in_Q1'] = 1 if 'Q1' in quadrants_present_in else 0
    df.loc[index,'text_in_Q2'] = 1 if 'Q2' in quadrants_present_in else 0
    df.loc[index,'text_in_Q3'] = 1 if 'Q3' in quadrants_present_in else 0
    df.loc[index,'text_in_Q4'] = 1 if 'Q4' in quadrants_present_in else 0
    df.loc[index,'area_covered_by_text'] =  round(area_covered,0)
    df.loc[index,'area_covered_text_of_image'] =  percent_covered
    df.loc[index,'word_count_image_text'] =  len(image_text.split(' '))
    df.loc[index,'avg_area_covered_by_a_word'] = round(area_covered/len(image_text.split(' ')),round_to_digits) if len(image_text.split(' ')) else 0
    df.loc[index,'img_text_str_len'] =  len(image_text)
    
    im.save(text_output_filename)

    df.loc[index,'chyron_present'] = get_chyron_band(img_path)
    
    
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 50)
    try:
        logos_detected = dill.load(file = open(responses_folder+'logos/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        logos_detected = [json.loads(MessageToJson(logo)) for logo in detect_logos(img_path)]
        dill.dump(logos_detected, file = open(responses_folder+'logos/{}.pkl'.format(img_id),'wb')) 
        
    area_covered_by_logo = 0
    logo_count = 0
    logo_in_title=list()
    quadrants_present_in =list()
    for n,logo in enumerate(logos_detected): 
        if logo['score'] < logo_detection_threshold : continue
        try:
            logo_box = ([(vertex['x'], vertex['y'])
                        for vertex in logo['boundingPoly']['vertices']])
        except:
            continue

        draw.line(logo_box + [logo_box[0]], width=5, fill='#00ff00')
            
        draw.text((logo_box[0][0],logo_box[0][1] - 30),
                  text= logo['description'],
                  fill='#000000',font=ImageFont.truetype('arial.ttf', 20))
        area_covered_by_logo += get_percent_covered(im,logo_box)
        quadrants_present_in.append(get_quadrant(im, logo_box))
        df2.loc[index,'logo_'+logo['description']] = 1
        if str(logo['description']).lower() in title.lower() : logo_in_title.append(1)
        logo_count+=1
    
    quadrants_present_in = [q for quadrant in list(dict.fromkeys(quadrants_present_in)) for q in quadrant.split(',')]
    quadrants_present_in = [q for q in list(dict.fromkeys(quadrants_present_in))] 
    df.loc[index,'logo_in_Q1'] = 1 if 'Q1' in quadrants_present_in else 0
    df.loc[index,'logo_in_Q2'] = 1 if 'Q2' in quadrants_present_in else 0
    df.loc[index,'logo_in_Q3'] = 1 if 'Q3' in quadrants_present_in else 0
    df.loc[index,'logo_in_Q4'] = 1 if 'Q4' in quadrants_present_in else 0
    
    df.loc[index,'area_covered_by_logo'] = round(area_covered_by_logo,0)
    df.loc[index,'logo_in_title'] = sum(logo_in_title)
    df.loc[index,'logo_count'] = logo_count
    df.loc[index,'avg_area_covered_per_logo'] = round(area_covered_by_logo/logo_count, round_to_digits) if logo_count else 0
    
    im.save(logo_output_filename)
    
    im = Image.open(img_path)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 50)

    try:
        objects_detected = dill.load(file = open(responses_folder+'objects/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        objects_detected = [json.loads(MessageToJson(object)) for object in object_localization(img_path) ]
        dill.dump(objects_detected, file = open(responses_folder+'objects/{}.pkl'.format(img_id),'wb'))
        
    area_covered_by_objects = 0
    objects_in_title=list()
    object_count = 0

    for object in objects_detected:
        if object['score'] < object_detection_threshold : continue 
        try:
            object_box = ([(denormalize_vertices(vertex['x'],im.size[0]), 
                            denormalize_vertices(vertex['y'],im.size[1])) 
                            for vertex in object['boundingPoly']['normalizedVertices']])
        except KeyError:
            continue
    
        draw.line(object_box + [object_box[0]], width=5, fill='#00ff00')
        draw.text((object_box[0][0],object_box[0][1] - 30),
              text= object['name'],
              fill='#000000',font=ImageFont.truetype('arial.ttf', 20))
              
        if str(object['name']).lower() in title.lower() : objects_in_title.append(1)
        
        area_covered_by_objects += get_percent_covered(im,object_box)
        mtv_wild_n_out2.loc[index,'object_'+object['name']] = 1
        
        if str(object['name']).lower() in title.lower(): df.loc[index,'object_in_title'] = 1
        object_count+=1
    
    df.loc[index,'area_covered_by_objects'] = round(area_covered_by_objects, 0)
    df.loc[index,'object_count'] = object_count
    df.loc[index,'avg_area_cvg_objects'] = round(area_covered_by_objects/object_count,round_to_digits) if object_count else 0
    df.loc[index,'objects_in_title'] = sum(objects_in_title)
    
    im.save(objects_output_filename)
    
    try:
        entities_detected = dill.load(file = open(responses_folder+'web_entities/{}.pkl'.format(img_id),'rb'))
    except FileNotFoundError:
        entities_detected= json.loads(MessageToJson(web_detection(img_path)))
        dill.dump(entities_detected, file = open(responses_folder+'web_entities/{}.pkl'.format(img_id),'wb'))
    
    talent_in_text=list()
    talent_count=0
    for entity in entities_detected['webEntities']:
        if not entity.get('score','') : continue
        if entity['score'] < web_detection_threshold: continue
        df2.loc[index,'talent_'+entity['description']] = 1
        talent_in_text.append(sum([1 for word in str(entity['description']).split() if word.lower() in title.lower()]))
        talent_count+=1
        
    
    df.loc[index,'title_text_num_talent'] = 1 if sum(talent_in_text)>1 else 0
    df.loc[index,'talent_count'] = talent_count
#    mtv_wild_n_out.loc[index,'title_text_talent_match_thumb_pos'] = 

df_final = pandas.merge(df,df2, how='left' , on ='video_id')
df_final.to_excel(output_directory+'CTR3and7_WNO_features.xlsx',index=False)
    
