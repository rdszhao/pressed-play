"""
URL configuration for spotapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path
from django.views.generic.base import RedirectView
from userflow import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('callback/', views.callback, name='callback'),
    path('genres/', views.get_genres, name='get_genres'),
    path('mainpage/', views.mainpage, name='mainpage'),
    path('infer/', views.infer_image, name='infer_image'),
    path('shuffle_genres/', views.shuffle_genres, name='shuffle_genres'),
    path('save_playlist/', views.save_playlist, name='save_playlist'),
    path('feedback/', views.feedback, name='feedback'),
    re_path(r'^$', RedirectView.as_view(url='login/')),
]