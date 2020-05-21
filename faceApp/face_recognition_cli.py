# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np

import json
from collections import OrderedDict

def upload_pictures(image_file, user_id, mode): #mode="photopost", "selfiepost"
    # 사진을 분석
    img = face_recognition.load_image_file(image_file)

    if (max(img.shape) > 1600):
        pil_img = PIL.Image.fromarray(img)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)  # 크기 줄임
        img = np.array(pil_img)

    user_encodings = face_recognition.face_encodings(img)

    # TODO. upload_encodings 실패시 예외처리 추가 , jpeg의 경우 인코딩이 안되는 경우 종종 발생. 확인 필요


    # user_id path 처리
    upload_name = user_id.username

    if(mode == "photopost"):
        file_path = "./media/unknown/unknown_encodings_save.json"
    else:
        if len(user_encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(image_file))
        if len(user_encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(image_file))

        file_path="./media/known/" + upload_name + "/known_encodings_save.json"

    # (기존)
    # if(not os.path.isdir("./media/images")): #처음 실행될 때
    #     upload_data = {}
    #     upload_data["unknowns"] = []
    # else:
    #     with open("unknown_encodings_save.json", "r") as f:
    #         upload_data = json.load(f)

    # NOTE. (수정)
    # 기존 - 첫 인코딩을 flag 파라미터를 받아서 판단.
    # 수정 - 인코딩 파일 존재여부로 판단, json 저장위치 변경

    if (not os.path.isfile(file_path)):
        upload_data = {};
        upload_data["photo"] = [];
    else:
        with open(file_path, "r") as f:
            upload_data = json.load(f);

    # numpy 를 array 로 변환
    upload_encodings = np.array(user_encodings)
    image_file_name = "./" + image_file.name
    upload_data["photo"].append({"name": image_file_name, "encodings": upload_encodings.tolist()})

    # python 'with'는 파일을 다룰 때 사용
    # 파일을 오픈하고 json_file 로 alias, .dump() 은 json을 해당 파일포인터로 파싱
    with open(file_path, "w", encoding="utf=8") as json_file:
        json.dump(upload_data, json_file, ensure_ascii=False, indent="\t")

#def compare_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
def compare_image(user_id, tolerance=0.4, show_distance=False):
    # 유저의 얼굴이 포함된 사진 이름 리스트
    user_photos = []
    known_file_path="./media/known/" + user_id + "/known_encodings_save.json"

    with open(known_file_path, "r") as known_file:
        known_data = json.load(known_file)

    for known in known_data['photo']:
        known_encodings = np.array(known['encodings'])

    unknown_file_path = "./media/unknown/unknown_encodings_save.json"

    with open(unknown_file_path, "r") as unknown_file:
        unknown_data = json.load(unknown_file)

    for unknown in unknown_data['photo']:
        unknown_encodings = np.array(unknown['encodings'])
        number_of_people = unknown_encodings.shape[0]# 사진에 몇명이 나왔는 지 확인

        if(number_of_people==1): # 사진 속 사람이 한 명일 경우
            distances = face_recognition.face_distance(known_encodings, unknown_encodings)
            result = list(distances <= tolerance)

            if True in result:
                user_photos.append(unknown['name'])

        else: # 사진 속에 2명 이상의 사람이 있을 경우
            for unknown_encoding in unknown_encodings:
                distances = face_recognition.face_distance(known_encodings, unknown_encoding)
                result = list(distances <= tolerance)

                if True in result:
                    user_photos.append(unknown['name'])
                    continue

    #중복 제거
    user_photos = list(set(user_photos))

    return user_photos

def image_files_in_folder(folder): # pwd 효과
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)


@click.command()
@click.argument('known_people_folder')
@click.argument('image_to_check')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def main(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    # known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1
"""
    if os.path.isdir(image_to_check):
        if cpus == 1:
            [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files_in_folder(image_to_check)]
        else:
            process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)

"""
if __name__ == "__main__":
    main()
