from django.shortcuts import render, redirect
from .models import Photo, Selfie
from .forms import PhotoPost, SelfiePost, PhotosPost
from django.utils import timezone
from django.contrib import messages
from django.contrib.auth import get_user
from faceApp.connect import connectionTest as connect
from faceApp.face_recognition_cli import *
from itertools import chain

# Create your views here.
def home(request):
    messages.info(request, "home 화면입니다")
    return render(request, 'home2.html')

def gallery(request):
    photos = Photo.objects;
    photos = photos.order_by('-created_date');
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
        for _file in request.FILES.getlist('image'):
            request.FILES['image'] = _file

            form = PhotosPost(request.POST, request.FILES)
            # print(form.is_valid())
            # 사진 업로드 구현부
            if form.is_valid():
                post = form.save(commit=False)
                post.created_date = timezone.now()
                post.owner = request.user
                #NOTE. save() 처리 과정 중 model.py/unique_file_name() 실행
                post.save()
                form.save_m2m()

                try:
                    upload_unknown_file(post.image.file);
                except NotFoundFace as e:
                    print(e, post.image.file);
                    messages.error(request, "얼굴을 찾을 수 없습니다. 다시 시도해 주세요.")
                    #TODO. 인코딩이 제대로 되지 않았을 때 처리 로직 추가. (디비와 파일에 저장된 사진 다시 지우기 등등..)
                    return redirect('home');
        messages.info(request, "저장 성공!")    
        messages.info(request, "인코딩 성공!");
        return redirect('gallery')
    # 일반 요청시
    else :
        form = PhotosPost()
        return render(request, 'new.html', {"form": form})

#NOTE. 셀피 업로드 시 동작하는 함수
def selfiepost(request):
    curUser = get_user(request)
    # print(curUser)
    if curUser.is_anonymous:
        messages.warning(request, "셀카를 업로드하시기 전에 로그인해주세요!")
        return redirect('home')

    if request.method == "POST":
        check = Selfie.objects.filter(owner = curUser)
        if( check.count() != 0 ):
            # 사용자 계정으로 등록된 셀피 삭제. 셀피는 하나만 존재
            print("이미 selfie가 등록되어 있습니다.")
            check.delete()

        form = SelfiePost(request.POST, request.FILES)
        if form.is_valid():
            post = form.save(commit=False)
            post.owner = request.user
            post.save()
            curUserId = curUser.id
            selfies = Selfie.objects.all()
            userSelfies = selfies.filter(owner_id=curUserId)

            # faceApp
            try:
                selfie_upload_btn(post.image.file, post.owner);
                messages.info(request, "셀피 업로드 성공!")

            except MoreThanOneFaceFound as e:
                print(e, post.image.file);
                messages.error(request, "한명의 얼굴이 나오게 사진을 찍어주세요.")
                return redirect('home');
            except NotFoundFace as e:
                print(e, post.image.file);
                messages.error(request, "얼굴을 찾을 수 없습니다. 다시 시도해 주세요.")
                return redirect('home');

            # 사진 검출 함수 호출, faceApp 결과 파일명 받아서 화면에 띄워주기
            return redirect('detectphoto')
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

        #사진 검출 함수 호출, faceApp 결과 파일명 받아서 화면에 띄워주기
        file_path="./media/known/" + curUser.username + "/known_encodings_save.json"
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)
            print(type(json_data['unknowns']))
            print(type(json_data['unknowns'][0]))
            known_encodings = np.array(json_data['unknowns'][0]['encodings'])

        result_arr = compare_image(image_to_check=None, known_names=None, known_face_encodings=known_encodings, tolerance=0.4)

        photos = Photo.objects.none() #empty queryset
        photos = list(photos)

        for result in result_arr:
            filename = result.split("/unknown")

            # 추출된 사진 띄우기
            photo = Photo.objects.all()
            photo = photo.order_by('-created_date');
            photo = photo.filter(image="unknown"+ filename[-1]);
            photos = list(chain(photos, photo))

        return render(request, 'selfie_gallery.html', {"username":curUser.username,"photos":photos})
    return redirect('gallery')
