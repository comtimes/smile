from django.shortcuts import render, redirect
from .models import Photo, Selfie
from .forms import PhotoPost, SelfiePost
from django.utils import timezone
from django.contrib import messages
from django.contrib.auth import get_user
from faceApp.connect import connectionTest as connect
from faceApp.face_recognition_cli import *

# Create your views here.
def home(request):
    messages.info(request, "home 화면입니다")
    return render(request, 'home2.html')

def gallery(request):
    photos = Photo.objects
    return render(request, 'gallery.html', {"photos": photos})

#NOTE. 사진 전송 시 동작하는 함수
def photopost(request):
    curUser = get_user(request)
    # print(curUser)
    # 로그인이 되지 않은 경우
    if curUser.is_anonymous:
        messages.warning(request, "사진을 업로드하시기 전에 로그인해주세요!")
        return redirect('home')

    # 사진 전송 버튼 클릭 시 동작하는 부분
    if request.method == 'POST':
        form = PhotoPost(request.POST, request.FILES)
        # print(form.is_valid())
        # 사진 업로드 구현부
        if form.is_valid():
            post = form.save(commit=False)
            post.created_date = timezone.now()
            post.owner = request.user
            post.save()
            #NOTE. save() 처리 과정 중 model.py/unique_file_name() 실행
            messages.info(request, "저장 성공!")

            #NOTE. faceApp 구현 부
            # (기존)upload_unknown_file(post.image, post.owner, 0);
            # @params: 업로드 된 사진의 경로를 전달
            # faceApp/face_recognition_cli 로 제어 이동
            upload_pictures(post.image, post.owner, "photopost")
            messages.info(request, "인코딩 성공!");

            return redirect('gallery')
    # 일반 요청시
    else :
        form = PhotoPost()
        return render(request, 'new.html', {"form": form})

#NOTE. 셀피 업로드 시 동작하는 함수
def selfiepost(request):
    curUser = get_user(request)
    # print(curUser)
    if curUser.is_anonymous:
        messages.warning(request, "셀카를 업로드하시기 전에 로그인해주세요!")
        return redirect('home')

    if request.method == "POST":
        form = SelfiePost(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.owner = request.user
            post.save()

            curUserId = curUser.id
            selfies = Selfie.objects.all()
            userSelfies = selfies.filter(owner_id=curUserId)

            # faceApp
            upload_pictures(post.image, post.owner, "selfiepost")
            messages.info(request, "셀피 업로드 성공!");

            return render(request, 'selfie_gallery.html', {"usernamne": curUser.username, "userselfies": userSelfies})

    else :
        form = SelfiePost()
        return render(request, 'new.html', {"form": form })

#NOTE. 사진 검출 시 동작하는 함수
def detectphoto(request):
    curUser = get_user(request)
    # print(curUser)
    # 로그인이 되지 않은 경우
    if curUser.is_anonymous:
        messages.warning(request, "사진을 검출하시기 전에 로그인해주세요!")
        return redirect('home')
    else :
        curUserId = curUser.id
        # print(curUserId)
        selfies = Selfie.objects.all()
        userSelfies = selfies.filter(owner_id=curUserId)
        # print(userSelfies.len())
        # print(userSelfies.count())
        if userSelfies.count() < 1:
            messages.warning(request, "사진을 검출하려면 최소 한개 이상의 selfie를 등록해주셔야 합니다!")
            return redirect('home')

        # 사진들 속에서 유저의 얼굴이 나온 사진을 검출
        user_photos = compare_image(curUser.username, 0.4, False)

        return render(request, 'selfie_gallery.html', {"usernamne":curUser.username,"userselfies":userSelfies})
    return redirect('gallery')
