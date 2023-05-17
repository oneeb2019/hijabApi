# views.py
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import requests
import cv2
import mediapipe as mp
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
import numpy as np
from flask import Flask, request, jsonify

# shirt = cv2.imread(r"C:\Users\Falcon\Downloads\HIJAB\1 (7).png", -1)
@csrf_exempt
def image_api(request):
    if request.method == 'POST':
        img = request.FILES['image']
        param = request.POST['mask']
        img_data = img.read()
        if param ==1:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\1.png", -1)
        if param ==2:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\2.png", -1)
        if param ==3:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\3.png", -1)
        if param ==4:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\4.png", -1)
        if param ==5:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\5.png", -1)
        if param ==6:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\6.png", -1)
        else:
            shirt = cv2.imread(r"Z:\Hajib_API\Hajib_API\API\HIJAB\1.png", -1)
        if img.content_type.split('/')[0] != 'image':
            return JsonResponse({'error': 'Invalid file type'})
        
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': img_data},
            data={'size': 'auto'},
            headers={'X-Api-Key': 'A3hQeQJaZgPGUgZgYUSNvQWp'},#GBXCUxogy3QttvmQGbpM2Sj1
        )
        
        if response.status_code == requests.codes.ok:
            with open('no-bg.png', 'wb') as out:
                out.write(response.content)
            
            # Read the image data from the output file
            with open('no-bg.png', 'rb') as f:
                img_data = f.read()
                
            # Encode the image data as Base64
            
            img = cv2.imread('no-bg.png')

            # Convert the image to RGB
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect the human body in the image
            mp_pose = mp.solutions.pose.Pose()

# Detect the human body in the image
            result1 = mp_pose.process(img_rgb)

            # Extract the head coordinates
            if result1.pose_landmarks:
                head_x = result1.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].x * img.shape[1]

                # Set the desired width for the cropped image
                crop_width = 500

                # Calculate the left and right coordinates for the crop
                crop_left = int(head_x - crop_width / 2)
                crop_right = int(head_x + crop_width / 2)

                # Ensure the crop stays within the image bounds
                if crop_left < 0:
                    crop_left = 0
                    crop_right = crop_width
                elif crop_right > img.shape[1]:
                    crop_right = img.shape[1]
                    crop_left = crop_right - crop_width

                # Crop the image to the desired width
                crop_img = img[:, crop_left:crop_right]

            
                # image = cv2.imread(crop_img)
                image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

                # Initialize the Mediapipe Pose Detection module
                mp_pose = mp.solutions.pose

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing = mp.solutions.drawing_utils    


                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    results = pose.process(image)

                    # Draw the pose landmarks and get the coordinates of the shoulder and hip joints
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        left_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1])
                        left_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
                        right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1])
                        right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])
                        left_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.shape[1])
                        left_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.shape[0])
                        right_hip_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.shape[1])
                        right_hip_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.shape[0])
                        # Get the coordinates of the left and right elbows
                        # Get the coordinates of the left and right shoulders
                        left_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1])
                        left_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0])
                        right_shoulder_x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1])
                        right_shoulder_y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0])

                        # Adjust the size of the shirt to fit the distance between the shoulders
                        shirt_width = right_shoulder_x - left_shoulder_x
                        shirt_height = int((right_shoulder_y - left_shoulder_y) * 1.5)
                        max_scale_factor = min(float(image.shape[1]) / shirt_width, float(image.shape[0]) / shirt_height)
                        if max_scale_factor <= 0:
                            # Resize the shirt image to be smaller than the image size
                            new_shirt_width = int(image.shape[1] / 1)
                            new_shirt_height = int(shirt.shape[0] *1.8)
                            shirt_resized = cv2.resize(shirt, (new_shirt_width, new_shirt_height), interpolation=cv2.INTER_AREA)
                            # Calculate the new scale factor
                            shirt_width = right_shoulder_x - left_shoulder_x
                            shirt_height = int((right_shoulder_y - left_shoulder_y) * 1.5)
                            max_scale_factor = min(float(image.shape[1]) / shirt_width, float(image.shape[0]) / shirt_height)
                        else:
                            shirt_resized = cv2.resize(shirt, None, fx=max_scale_factor, fy=max_scale_factor, interpolation=cv2.INTER_AREA)

                        # Overlay the shirt on the image, covering the shoulders
                        # Overlay the shirt on the body
                        right_shoulder_x = right_shoulder_x - 150
                        right_shoulder_y = right_shoulder_y -  95


                        for i in range(shirt_resized.shape[0]):
                            for j in range(shirt_resized.shape[1]):
                                if right_shoulder_x+j >= 0 and right_shoulder_x+j < image.shape[1] and right_shoulder_y+i >= 0 and right_shoulder_y+i < image.shape[0]:
                                    if shirt_resized[i, j, 3] != 0:
                                        image[right_shoulder_y+i, right_shoulder_x+j, :] = shirt_resized[i, j, :3]
                                        
                                        
                        retval, buffer = cv2.imencode('.jpg', image)
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                        return JsonResponse({"Hijab":jpg_as_text})
            
            return JsonResponse({'error': 'No human body detected in the image..'})
        
            
            

        else:
            return JsonResponse({"Error:": response.status_code,"LOG": response.text})
    else:
        return JsonResponse({'error': 'Invalid request method'})
import io
@csrf_exempt
def image_api_v2(request):
    if request.method == 'POST':
        param = request.POST['mask']
        img_file = request.FILES['image']
        print(type(param))
        print(param)
        
        mp_pose = mp.solutions.pose.Pose()
        # a="output_image.jpg"
        if param =="1":
            a=r"C:\Users\Falcon\hijabApi\API\HIJAB\1.png"
            paste_x =90
            paste_y = 185
        elif param == "2":
            a = r"C:\Users\Falcon\hijabApi\API\HIJAB\2.png"
            paste_x =90
            paste_y = 185
        elif param == "3":
            a = r"C:\Users\Falcon\hijabApi\API\HIJAB\3.png"
            paste_x =90
            paste_y = 170
        elif param == "4":
            a = r"C:\Users\Falcon\hijabApi\API\HIJAB\4.png"
            paste_x =80
            paste_y = 208
        elif param == "5":
            a = r"C:\Users\Falcon\hijabApi\API\HIJAB\5.png"
            paste_x =100
            paste_y = 178
        elif param == "6":
            a = r"C:\Users\Falcon\hijabApi\API\HIJAB\6.png"
            paste_x =95
            paste_y = 184
        elif param =="7":
            a=r"C:\Users\Falcon\hijabApi\API\HIJAB\7.png"
            paste_x =95
            paste_y = 150
            px=1.4
        else:
            return JsonResponse({'error':'Please select Maskin range 1-7 '})


        img_buffer = io.BytesIO(img_file.read())

        # Rewind the buffer to the beginning
        img_buffer.seek(0)

        # Load the image using OpenCV from the buffer
        img_cv = cv2.imdecode(np.frombuffer(img_buffer.read(), np.uint8), cv2.IMREAD_COLOR)
        # img_cv = cv2.resize(img_cv, (600, 600))
        img = cv2.resize(img_cv, (600, 600))

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect the human body in the image
        result = mp_pose.process(img_rgb)

        # Extract the head coordinates
        if result.pose_landmarks:
            head_x = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].x * img.shape[1]
            head_y = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE].y * img.shape[0]

            # Crop the image to focus on the head
            crop_margin = 100
            head_width = 100
            head_height = 100
            crop_img = img[int(head_y-head_height-crop_margin):int(head_y+crop_margin), int(head_x-head_width-crop_margin):int(head_x+head_width+crop_margin)]

            # Resize the cropped image
            crop_img = cv2.resize(crop_img, (250, 250))

            # Create a blank image
            blank_img = np.zeros((800, 500, 3), dtype=np.uint8)

            # Place the cropped image on top of the blank image
            h, w = crop_img.shape[:2]
            x = int((blank_img.shape[1] - w) / 2)
            y = 0
            blank_img[y:y+h, x:x+w] = crop_img

            # Save the result image
            output_path = 'output_image.jpg'
            cv2.imwrite(output_path, blank_img)

            # Load the base image and the image to paste
            base_image = Image.open(output_path)
            paste_image = Image.open(a)

            base_width, base_height = base_image.size

            # Resize the paste image to fit within the base image
            px=1.2
            print("before")
            paste_image = paste_image.resize((int(base_width / 1.4), int(base_height / px)))
            print("after")
            

            # Define the position to paste the image
            # paste_x =80
            # paste_y = 208

            # Paste the image onto the base image
            base_image = base_image.convert("RGBA")
            paste_image = paste_image.convert("RGBA")
            base_image.paste(paste_image, (paste_x, paste_y), paste_image)

# Convert the result to BGR format
            result = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGBA2BGR)

            # Save the result image to a file
            cv2.imwrite('result.png', result)

            # Make the API call
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files={'image_file': open('result.png', 'rb')},
                data={'size': 'auto'},
                headers={'X-Api-Key': 'cozRq3K4ReJgYsLYBd2zwJmn'},
            )

            # Check the response status code
            if response.status_code == requests.codes.ok:
                # Save the output image
                with open('no-bg.png', 'wb') as out:
                    out.write(response.content)
                with open('no-bg.png', 'rb') as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                return JsonResponse({"Hijab": encoded_image})
            else:
                return JsonResponse({'error': 'BG remover not working'})

                
            
            
        else:
            return JsonResponse({'error': 'No human body detected in the image.'})
        
    
    
    
    
    
    
    
    
    
    
