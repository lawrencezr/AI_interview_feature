import os
import json
import math
import pandas as pd


def load_frame_json(file_folder):
    files = os.listdir(file_folder)
    file_list = []
    for path in files:
        file_path = os.path.join(file_folder,path)
        file_list.append(file_path)
    result = []
    for file_path in file_list:
        with open(file_path, 'rb') as file:
            print(file_path)
            result.append(json.load(file))
            # print(frame['images'file][0])
            # frames = result['images']
    num = len(result)  # 视频数
    # print(result[:2])
    # print(num)
    return result,num

# 颜值
def beauty(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    beauty = []
    for i in range(0,numVideo):
        frames = result[i]['images']
        video_id = result[i]['video_info']['question_id']
        numFrame = len(frames)
        beautyList=[]
    # print(frames[0]['image']['face']['beauty']['score'])
    # print(len(frames))
    #     print(frames[0]['image']['face']['beauty']['score'])
    #     print('video_id', video_id)
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j]['image']['face'] and frames[j]['image']['face']['beauty']!=[]:
                beautyList.append(frames[j]['image']['face']['beauty']['score'])
        beautyList.sort(reverse=True)
        # print(beautyList)
        if len(beautyList)==0:
            beauty.append('NaN')
        elif len(beautyList)==1:
            beauty.append(round(beautyList[0],2))
        elif len(beautyList)==2:
            beauty.append(round((beautyList[0] + beautyList[1]) / 2.0,2))
        else:
            beauty.append(round((beautyList[1]+beautyList[2])/2.0,2))
    print('beauty:', beauty)
    return beauty

# 面色状态
def skin_status(result,numVideo):
    # result,numVideo= load_frame_json('result/')
    darkCircle = []
    stain = []
    acne = []
    health = []
    for i in range(0,numVideo):
        video_id = result[i]['video_info']['question_id']
        frames = result[i]['images']
        numFrame = len(frames)
        num = numFrame# 帧数
        sumDarkCircle = 0
        sumStain = 0
        sumAcne = 0
        sumHealth = 0
        # print('video_id', video_id)
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j]['image']['face'] and \
                    frames[j]['image']['face']['skinstatus']!= []:
                sumDarkCircle += frames[j]['image']['face']['skinstatus']['dark_circle']
                sumStain += frames[j]['image']['face']['skinstatus']['stain']
                sumAcne += frames[j]['image']['face']['skinstatus']['acne']
                sumHealth += frames[j]['image']['face']['skinstatus']['health']
            else:
                num -= 1
        if num != 0:
            sumDarkCircle /= num
            sumStain /= num
            sumAcne /= num
            sumHealth /= num
            darkCircle.append(round(sumDarkCircle,2))
            stain.append(round(sumStain,2))
            acne.append(round(sumAcne,2))
            health.append(round(sumHealth,2))
        else:
            darkCircle.append('NaN')
            stain.append('NaN')
            acne.append('NaN')
            health.append('NaN')
        # print(i)
        # print('darkCircle: ', sumDarkCircle)
        # print('stain: ', sumStain)
        # print('acne: ', sumAcne)
        # print('health: ', sumHealth)
    print('darkCircle: ' , darkCircle)
    print('stain: ' , stain)
    print('acne: ' , acne)
    print('health: ' ,health)
    return darkCircle, stain, acne, health

# 面部遮挡
def face_occlusion(result,numVideo):
    # result,numVideo = load_frame_json('frame_result.json')
    # 0无遮挡 1有遮挡
    left_eye = []
    right_eye = []
    nose = []
    mouth = []
    left_cheek = []
    right_cheek = []
    chin_contour = []
    # 阈值为API推荐阈值
    for i in range(0,numVideo):
        frames = result[i]['images']
        numFrame = len(frames)
        row_left_eye=[]
        row_right_eye=[]
        row_nose=[]
        row_mouth=[]
        row_left_cheek=[]
        row_right_cheek=[]
        row_chin_contour=[]
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j]['image']['face']:
                if frames[j]['image']['face']['face_quality']['occlusion']['left_eye']>0.6:
                    row_left_eye.append(1)
                else:
                    row_left_eye.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['right_eye']>0.6:
                    row_right_eye.append(1)
                else:
                    row_right_eye.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['nose']>0.7:
                    row_nose.append(1)
                else:
                    row_nose.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['mouth']>0.7:
                    row_mouth.append(1)
                else:
                    row_mouth.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['left_cheek']>0.8:
                    row_left_cheek.append(1)
                else:
                    row_left_cheek.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['right_eye']>0.8:
                    row_right_cheek.append(1)
                else:
                    row_right_cheek.append(0)
                if frames[j]['image']['face']['face_quality']['occlusion']['chin_contour']>0.6:
                    row_chin_contour.append(1)
                else:
                    row_chin_contour.append(0)
        left_eye.append(row_left_eye)
        right_eye.append(row_right_eye)
        nose.append(row_nose)
        mouth.append(row_mouth)
        left_cheek.append(row_left_cheek)
        right_cheek.append(row_right_cheek)
        chin_contour.append(row_chin_contour)
    print('left_eye: ',left_eye)
    print('right_eye: ' , right_eye)
    print('nose: ' , nose)
    print('mouth: ' , mouth)
    print('left_cheek: ' , left_cheek)
    print('right_cheek: ' , right_cheek)
    print('chin_contour: ' , chin_contour)
    return left_eye, right_eye, nose, mouth, left_cheek, right_cheek, chin_contour

# 眨眼次数
def blink(result,numVideo):
    # frames,numVideo = load_frame_json('frame_result.json')
    blink = 0
    leftEyeList = []
    rightEyeList = []
    for i in range(0,numVideo):
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['normal_glass_eye_open'])
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['no_glass_eye_close'])
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['occlusion'])
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['no_glass_eye_open'])
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['normal_glass_eye_close'])
        # leftEyeList.append(frames[i]['image']['face']['eyestatus']['left_eye_status']['dark_glasses'])
        # leftEyeList.sort(reverse=True)
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['normal_glass_eye_open'])
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['no_glass_eye_close'])
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['occlusion'])
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['no_glass_eye_open'])
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['normal_glass_eye_close'])
        # rightEyeList.append(frames[i]['image']['face']['eyestatus']['right_eye_status']['dark_glasses'])
        # rightEyeList.sort(reverse=True)
        leftEye = frames[i]['image']['face']['eyestatus']['left_eye_status']
        rightEye = frames[i]['image']['face']['eyestatus']['right_eye_status']
        leftEyeList = sorted(leftEye.items(),key=lambda x:x[1],reverse=True)
        rightEyeList = sorted(rightEye.items(),key=lambda x:x[1],reverse=True)
        if leftEyeList[0][0] == 'normal_glass_eye_close' or leftEyeList[0][0] =='no_glass_eye_close' or \
                rightEyeList[0][0] == 'normal_glass_eye_close' or rightEyeList[0][0] == 'no_glass_eye_close':
            blink += 1
        # print(leftEyeList[0][0])
        # print(rightEyeList[0][0])
        leftEyeList = []
        rightEyeList = []
    print('blink: %d'%blink)
    return blink

# 光线
def light(result,numVideo):
    # result,numVideo = load_frame_json('frame_result.json')
    # 光线良好为1 不好为0
    light = []
    for i in range(0,numVideo):
        frames = result[i]['images']
        numFrame = len(frames)
        num = numFrame
        rowLight=[]
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image']:
        # print(frames[i]['image']['face']['face_quality']['illumination'])
                if frames[j]['image']['face']['face_quality']['illumination'] > 40:
                    rowLight.append(1)
                else:
                    rowLight.append(0)
            # else:
            #     num -= 1
        # if numLight >= num/3*4:
        #     light[i] = 1
        light.append(rowLight)
    print('light: ',light)
    return light

# 衣着
def clothes(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    upperWearList = []
    suit = [] #正装1 非正装0
    cap = [] #戴帽子1 不戴0
    mask = [] #戴口罩1 不戴0
    blackGlasses = [] #戴墨镜1 不戴0
    colorList = []
    textureList = []
    for i in range(0,numVideo):
        video_id = result[i]['video_info']['question_id']
        # print('video_id',video_id)
        frames = result[i]['images']
        suit.append(-1)
        numFrame = len(frames)
        upperWear = {'长袖': 0, '短袖': 0}  # 长袖1 短袖0
        rowSuit = []
        rowCap = []  # 戴帽帧数
        rowMask = []
        rowBg = []
        color = {'红': 0, '橙': 0, '黄': 0, '绿': 0, '蓝': 0, '紫': 0, '粉': 0, '黑': 0, '白': 0, '灰': 0, '棕': 0}  # 衣服颜色
        texture = {'纯色': 0, '图案': 0, '碎花': 0, '条纹或格子': 0}  # 衣服纹理
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j]['image']['face']:
                upperWear[frames[j]['image']['body']['attributes']['upper_wear']['name']] += 1
                if frames[j]['image']['body']['attributes']['upper_wear_fg']['name'] == '西装':
                    # rowSuit.append(1)
                    suit[i]=1
                if frames[j]['image']['body']['attributes']['headwear']['name'] != '无帽':
                    rowCap.append(1)
                else:
                    rowCap.append(0)
                mouth = frames[j]['image']['face']['mouthstatus']
                if mouth != []:
                    mouthList = sorted(mouth.items(),key=lambda x:x[1],reverse=True)
                    if mouthList[0][0] == 'surgical_mask_or_respirator':
                        rowMask.append(1)
                    else:
                        rowMask.append(0)
                if frames[j]['image']['body']['attributes']['glasses']['name'] == '戴墨镜':
                    rowBg.append(1)
                else:
                    rowBg.append(0)
                color[frames[j]['image']['body']['attributes']['upper_color']['name']] += 1
                texture[frames[j]['image']['body']['attributes']['upper_wear_texture']['name']] += 1
                # print('j:',j)
            else:
                rowSuit.append('error')
                rowBg.append('error')
                rowMask.append('error')
                rowCap.append('error')
        # suit.append(rowSuit)
        cap.append(rowCap)
        mask.append(rowMask)
        blackGlasses.append(rowBg)
        upperWearList.append(upperWear)
        colorList.append(color)
        textureList.append(texture)
        # print('i:',i)
    print('upperWear:', upperWearList)
    print('suit: ' ,suit)
    print('cap: ' , cap)
    print('mask: ' , mask)
    print('blackGlasses: ' , blackGlasses)
    print('color: ', colorList)
    print('texture: ', textureList)
    return suit

#是否有手势
def has_gesture(result,numVideo):
    # result,numVideo = load_frame_json('frame_result.json')
    hasGesture = []
    for i in range(0,numVideo):
        frames = result[i]['images']
        numFrame = len(frames)
        rowGesture = []
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image'] and frames[j]['image']['gesture']!=[]:
                for k in frames[j]['image']['gesture']:
                    if k['classname'] != 'Face':
                        rowGesture.append(1)
                        break
                    else:
                        rowGesture.append(0)
        hasGesture.append(rowGesture)
    print('hasGesture: ',hasGesture)
    return hasGesture

# 头部晃动角度
def head_angle(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    yawHead = []
    pitchHead = []
    rollHead = []
    for i in range(0,numVideo):
        video_id = result[i]['video_info']['question_id']
        # print('video_id',video_id)
        frames = result[i]['images']
        numFrame = len(frames)
        sumYaw = 0
        sumPitch = 0
        sumRoll = 0
        for j in range(0,numFrame-1):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j+1]['image'] and \
                'error' not in frames[j]['image']['face'] and 'error' not in frames[j+1]['image']['face']:
                sumYaw += abs(float(frames[j]['image']['face']['headpose']['yaw_angle'])-float(frames[j+1]['image']['face']['headpose']['yaw_angle']))
                sumPitch += abs(float(
                    frames[j]['image']['face']['headpose']['pitch_angle']) - float(frames[j + 1]['image']['face']['headpose'][
                        'pitch_angle']))
                sumRoll += abs(float(
                    frames[j]['image']['face']['headpose']['roll_angle']) - float(frames[j + 1]['image']['face']['headpose'][
                        'roll_angle']))
        yawHead.append(round(sumYaw,2))
        pitchHead.append(round(sumPitch,2))
        rollHead.append(round(sumRoll,2))
    print('yawHead: ', yawHead)
    print('pitchHead: ', pitchHead)
    print('rollHead: ', rollHead)
    return yawHead, pitchHead, rollHead

#身体晃动
def body_shake(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    bodyShake = []
    for i in range(0,numVideo):
        video_id = result[i]['video_info']['question_id']
        # print('video_id',video_id)
        frames = result[i]['images']
        numFrame = len(frames)
        sumShake = 0
        for j in range(0,numFrame-1):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j + 1]['image'] and \
                    'error' not in frames[j]['image']['face'] and 'error' not in frames[j + 1]['image']['face'] and \
                    (frames[j]['image']['skeleton']['body_parts']['left_shoulder']['x'] != 0.0 and \
                    frames[j+1]['image']['skeleton']['body_parts']['left_shoulder']['x'] != 0.0):
                sumShake += math.sqrt(pow(frames[j]['image']['skeleton']['body_parts']['left_shoulder']['x'] -
                                          frames[j+1]['image']['skeleton']['body_parts']['left_shoulder']['x'],2) +
                                      pow(frames[j]['image']['skeleton']['body_parts']['left_shoulder']['y'] -
                                          frames[j + 1]['image']['skeleton']['body_parts']['left_shoulder']['y'], 2)
                                      )
        bodyShake.append(round(sumShake,2))
    print('bodyShake: ',bodyShake)
    return bodyShake

#情绪多样性
def emotion_diversity(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    emotionDiversity = []
    for i in range(0,numVideo):
        frames = result[i]['images']
        numFrame = len(frames)
        emotionDic = {'sadness':0,'neutral':0,'disgust':0,'anger':0,'surprise':0,'fear':0,'happiness':0, 'angry':0, 'happy':0,'sad':0}
        for j in range(0,numFrame):
            if 'error' not in frames[j]['image']:
                emotion = frames[j]['image']['face']['emotion']
                emotionList = sorted(emotion.items(),key=lambda x:x[1],reverse=True)
                if emotionList[0][0] in emotionDic:
                    emotionDic[emotionList[0][0]]+=1
        # print(i,': ',emotionDic)
        emotionDiversity.append(emotionDic)
    print('emotionDiversity: ', emotionDiversity)
    return emotionDiversity

def smile(result,numVideo):
    # result,numVideo = load_frame_json('result/')
    smile = []
    for i in range(numVideo):
        frames = result[i]['images']
        numFrame = len(frames)
        num = numFrame
        sumSmile = 0.0
        for j in range(numFrame):
            if 'error' not in frames[j]['image'] and 'error' not in frames[j]['image']['face']:
                sumSmile += frames[j]['image']['face']['smile']['value']
            else:
                num -= 1
        if num == 0:
            smile.append('NaN')
        else:
            smile.append(sumSmile/num)
    print('smile: ',smile)
    return smile



if __name__ == '__main__':
    result,numVideo= load_frame_json('result/')
    videoList = []
    for i in range(0, numVideo):
        videoList.append(result[i]['video_info']['question_id'])
    beauty = beauty(result,numVideo)
    darkCircle, stain, acne, health = skin_status(result,numVideo)
    # # face_occlusion()
    # # # blink()
    # # light()
    # # clothes()
    # # has_gesture()
    yawHead, pitchHead, rollHead = head_angle(result,numVideo)
    bodyShake = body_shake(result,numVideo)
    suit = clothes(result,numVideo)
    smile = smile(result,numVideo)
    # # emotion_diversity()
    df = pd.DataFrame({'id':videoList,'beauty':beauty, 'darkCircle':darkCircle,
                       'stain':stain, 'acne':acne,'health':health, 'suit':suit,'smile':smile,'yawHead':yawHead,
                       'pitchHead':pitchHead, 'rollHead':rollHead,'bodyShake':bodyShake})
    df.to_csv('result_compare.csv',index=False,columns=['id','beauty','darkCircle','stain','acne','health','suit','smile',
                                                'yawHead','pitchHead','rollHead','bodyShake'])

    # columes = ['id','frame_1','frame_2','frame_3','frame_4','frame_5','frame_6','frame_7',
    #            'frame_8','frame_9','frame_10']
    # for i in range(0,len(suit)):
    #     suit[i].insert(0,videoList[i])
    # df_clothes = pd.DataFrame(columns=columes,data=suit)
    # df_clothes.to_csv('result_clothes.csv',index=False)
    # result,num = load_frame_json('result/')