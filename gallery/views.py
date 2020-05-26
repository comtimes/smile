from django.shortcuts import render, redirect
from .models import Photo, Selfie
from .forms import PhotoPost, SelfiePost
from django.utils import timezone
from django.contrib import messages
from django.contrib.auth import get_user

# Create your views here.
def home(request):
    messages.info(request, "home 화면입니다")
    return render(request, 'home2.html')

def gallery(request):
    photos = Photo.objects
    return render(request, 'gallery.html', {"photos": photos})

def photopost(request):
    curUser = get_user(request)
    # print(curUser)
    if curUser.is_anonymous:
        messages.warning(request, "사진을 업로드하시기 전에 로그인해주세요!")
        return redirect('home')

    if request.method == 'POST':
        form = PhotoPost(request.POST, request.FILES)
        # print(form.is_valid())
        if form.is_valid():
            post = form.save(commit=False)
            post.created_date = timezone.now()
            post.owner = request.user
            post.save()
            messages.info(request, "저장 성공!")
            return redirect('gallery')
    else :
        form = PhotoPost()
        return render(request, 'new.html', {"form": form})

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
            return redirect('gallery')
    else :
        form = SelfiePost()
        return render(request, 'new.html', {"form": form })

def detectphoto(request):
    curUser = get_user(request)
    # print(curUser)
    if curUser.is_anonymous:
        messages.warning(request, "사진을 검출하시기 전에 로그인해주세요!")
        return redirect('home')
    else :
        curUserId = curUser.id
        # print(curUserId)
        selfies = Selfie.objects.all()
        userSelfies = selfies.filter(owner_id=curUserId)
        # print(selfies)
        # print(userSelfies)
        # print(userSelfies.count())
        if userSelfies.count() < 1:
            messages.warning(request, "사진을 검출하려면 최소 한개 이상의 selfie를 등록해주셔야 합니다!")
            return redirect('home')
            
        return render(request, 'selfie_gallery.html', {"usernamne":curUser.username,"userselfies":userSelfies})
    return redirect('gallery')
