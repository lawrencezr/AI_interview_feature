import os
import json

def load_frame_json(filePath):
    with open(filePath,'r') as file:
        result = json.load(file)
        # print(frame['images'][0])
        frames = result['images']
        num = len(frames)
    return frames,num

# 颜值
def beauty():
    frames,num = load_frame_json('frame_feature.json')
    beautyList=[]
    # print(frames[0]['image']['face']['beauty']['score'])
    # print(len(frames))
    for i in range(0,num):
        beautyList.append(frames[i]['image']['face']['beauty']['score'])
    beautyList.sort(reverse=True)
    beauty = (beautyList[1]+beautyList[2])/2.0
    print('beauty: %.3f' % beauty)
    return beauty

# 面色状态
def skin_status():
    frames,num = load_frame_json('frame_feature.json')
    darkCircle = 0.0
    stain = 0.0
    acne = 0.0
    health = 0.0
    for i in range(0,num):
        darkCircle += frames[i]['image']['face']['skinstatus']['dark_circle']
        stain += frames[i]['image']['face']['skinstatus']['stain']
        acne += frames[i]['image']['face']['skinstatus']['acne']
        health += frames[i]['image']['face']['skinstatus']['health']
    darkCircle /= num
    stain /= num
    acne /= num
    health /= num
    print('darkCircle: %.3f' % darkCircle)
    print('stain: %.3f' % stain)
    print('acne: %.3f' % acne)
    print('health: %.3f' % health)
    return darkCircle, stain, acne, health

# 面部遮挡
def face_occlusion():
    frames,num = load_frame_json('frame_feature.json')
    # 0无遮挡 1有遮挡
    left_eye = 0
    right_eye = 0
    nose = 0
    mouth = 0
    left_cheek = 0
    right_cheek = 0
    chin_contour = 0
    # 阈值为API推荐阈值
    for i in range(0,num):
        if frames[i]['image']['face']['face_quality']['occlusion']['left_eye']>0.6:
            left_eye = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['right_eye']>0.6:
            right_eye = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['nose']>0.7:
            nose = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['mouth']>0.7:
            mouth = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['left_cheek']>0.8:
            left_cheek = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['right_eye']>0.8:
            right_cheek = 1
        if frames[i]['image']['face']['face_quality']['occlusion']['chin_contour']>0.6:
            chin_contour = 1
    print('left_eye: %d'%left_eye)
    print('right_eye: %d' % right_eye)
    print('nose: %d' % nose)
    print('mouth: %d' % mouth)
    print('left_cheek: %d' % left_cheek)
    print('right_cheek: %d' % right_cheek)
    print('chin_contour: %d' % chin_contour)
    return left_eye, right_eye, nose, mouth, left_cheek, right_cheek, chin_contour

# 眨眼次数
def blink():
    frames,num = load_frame_json('frame_feature.json')
    blink = 0
    leftEyeList = []
    rightEyeList = []
    for i in range(0,num):
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
def light():
    frames, num = load_frame_json('frame_feature.json')
    # 光线良好为1 不好为0
    light = 0
    numLight = 0
    for i in range(0,num):
        # print(frames[i]['image']['face']['face_quality']['illumination'])
        if frames[i]['image']['face']['face_quality']['illumination'] > 40:
            numLight+=1
    if numLight >= 7:
        light = 1
    else:
        light = 0
    print('light: %d'%light)
    return light

# 衣着
def clothes():
    frames, num = load_frame_json('frame_feature.json')
    upperWear = {'长袖':0,'短袖':0} #长袖1 短袖0
    suit = 0 #正装1 非正装0
    numSuit = 0
    cap = 0 #戴帽子1 不戴0
    numCap = 0 #无帽帧数
    mask = 0 #戴口罩1 不戴0
    numMask = 0
    blackGlasses = 0 #戴墨镜1 不戴0
    color = {'红':0,'橙':0,'黄':0,'绿':0,'蓝':0,'紫':0,'粉':0,'黑':0,'白':0,'灰':0,'棕':0} #衣服颜色
    texture = {'纯色':0,'图案':0,'碎花':0,'条纹':0,'格子':0} #衣服纹理
    for i in range(0,num):
        upperWear[frames[i]['image']['body']['attributes']['upper_wear']['name']] += 1
        if frames[i]['image']['body']['attributes']['upper_wear_fg']['name'] == '西装':
            numSuit+=1
        if frames[i]['image']['body']['attributes']['headwear']['name'] == '无帽':
            numCap+=1
        mouth = frames[i]['image']['face']['mouthstatus']
        mouthList = sorted(mouth.items(),key=lambda x:x[1],reverse=True)
        if mouthList[0][0] == 'surgical_mask_or_respirator':
            numMask+=1
        if frames[i]['image']['body']['attributes']['glasses']['name'] == '戴墨镜':
        if mouthList[0][0]




if __name__ == '__main__':
    beauty()
    skin_status()
    face_occlusion()
    blink()
    light()